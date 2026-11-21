# rbf_server_app.py — Documentation

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Core Classes](#3-core-classes)
   - [FrontendState](#31-frontendstate)
   - [CudaRBFRuntime](#32-cudarbfruntime)
   - [RBFServerController](#33-rbfservercontroller)
   - [FullFieldRenderManager](#34-fullfieldrendermanager)
   - [RealtimeDecoderManager](#35-realtimedecodermanager)
4. [RBF Interpolation and Latent Field](#4-rbf-interpolation-and-latent-field)
   - [Radial Basis Functions](#41-radial-basis-functions)
   - [Latent Fallback Modes](#42-latent-fallback-modes)
   - [Blend Math](#43-blend-math)
   - [BPM Interpolation](#44-bpm-interpolation)
5. [Heatmap Visualization](#5-heatmap-visualization)
6. [Decoding: Latent → MIDI Matrix](#6-decoding-latent--midi-matrix)
7. [Model Checkpoint Management](#7-model-checkpoint-management)
8. [Communication Protocol (SocketIO)](#8-communication-protocol-socketio)
   - [Events](#81-events)
   - [Payload Structure (client → server)](#82-payload-structure-client--server)
   - [Response Structure (server → client)](#83-response-structure-server--client)
9. [UI and Visualization (template.html)](#9-ui-and-visualization-templatehtml)
10. [Full-Field Render Output Format (.npz)](#10-full-field-render-output-format-npz)
11. [Threading Model](#11-threading-model)
12. [Startup and Configuration](#12-startup-and-configuration)
13. [Training Tab](#13-training-tab)
14. [Dataset Tab](#14-dataset-tab)
15. [Compound Token Format](#15-compound-token-format)

---

## 1. Overview

`rbf_server_app.py` is the backend server for the **Pattern Atlas** interactive latent-space explorer. It exposes a browser-based UI (served from `template.html`) that lets the user drag anchor points around a 2-D field, with the server continuously interpolating a VAE latent vector at the cursor position using Radial Basis Function (RBF) interpolation. The resulting latent is decoded by a Transformer VAE into a drum pattern (MIDI matrix), and visualized in real time.

Key capabilities:

- Real-time RBF latent interpolation and BPM interpolation at the cursor position
- Configurable fallback behavior for low-influence / out-of-field regions
- Live Transformer VAE decoding of the cursor latent into a MIDI piano-roll matrix
- Full-field batch render: decode every cell in a `grid_resolution × grid_resolution` grid and save to `.npz`
- Heatmap overlay showing latent magnitude or anchor influence (confidence) over the field
- Hot-swappable model checkpoints (best_val, best_train, last, none) without server restart
- MIDI encode: encode a MIDI input back into the latent space via the VAE encoder

The server is built on **Flask + Flask-SocketIO** and communicates with the frontend over WebSocket using a frame-based request/response loop.

---

## 2. Architecture

```
Browser (template.html)
    │  WebSocket (SocketIO)
    │  emit("bootstrap") / emit("frame", payload)
    ▼
Flask-SocketIO Server
    │
    ├── RBFServerController          ← main coordinator, holds all state
    │       ├── CudaRBFRuntime       ← VAE model (encoder + decoder), device management
    │       ├── RealtimeDecoderManager  ← background thread: latent → MIDI matrix
    │       └── FullFieldRenderManager  ← background thread: full-field batch render
    │
    └── FrontendState                ← single dataclass mirroring all UI-controlled state
```

The `RBFServerController.process_frame()` method is the central hot path called on every `"frame"` SocketIO event. It:
1. Parses the incoming payload
2. Recomputes RBF weights if control points or sigma changed (signature-cached)
3. Interpolates the latent vector and BPM at the cursor position
4. Updates the heatmap cache if needed (signature-cached)
5. Triggers background decoding or rendering as appropriate
6. Returns a full response dict that is emitted back to the client

---

## 3. Core Classes

### 3.1 FrontendState

A `@dataclass` that holds the single source of truth for all UI-driven state on the server side. It is updated atomically inside the `_lock` on every `process_frame` call.

| Field | Type | Default | Description |
|---|---|---|---|
| `cursor` | `np.ndarray (2,)` | — | Current cursor position in field coordinates |
| `control_points` | `np.ndarray (N, 2)` | — | Anchor positions in field coordinates |
| `heatmap_mode` | `str` | `"magnitude"` | `"magnitude"` or `"confidence"` |
| `influence_mode` | `str` | `"raw"` | `"raw"` or `"percentage"` |
| `decode_enabled` | `bool` | `False` | Whether live decoding is active |
| `render_enabled` | `bool` | `False` | Whether full-field render is requested |
| `bpm_interp_mode` | `str` | `"gaussian_blend"` | `"gaussian_blend"` or `"exact_rbf"` |
| `latent_fallback_mode` | `str` | `"preserve_current_behavior"` | See §4.2 |
| `grid_resolution` | `int` | `30` | Grid cells per axis (4–60) |
| `sigma` | `float` | `0.30` | RBF kernel width |
| `latent_vector` | `np.ndarray \| None` | `None` | Last computed latent at cursor |
| `current_bpm` | `float` | `0.0` | Last interpolated BPM |
| `last_frontend_payload` | `dict \| None` | `None` | Echo of last received payload |
| `refined_blend_low` | `float` | `0.0002` | Influence threshold below which fallback is 100% active |
| `refined_blend_high` | `float` | `0.45` | Influence threshold above which RBF is 100% active |

---

### 3.2 CudaRBFRuntime

Manages model loading and device selection (CUDA → MPS → CPU). Wraps the `TransformerVAE` and exposes:

- `decode_latents_to_matrices(latents)` / `decode_latent_to_matrix(latent_vector)` — batched greedy, constrained decoding with KV-cache (`Decoder.generate`) into piano-roll matrices `(height × width*n_timing)`; the sequence ends at the first EOS frame
- `encode_midi_buffer_to_latent(midi_buffer)` — encodes `{frames: [[step, pitch, velocity, deviation], ...]}`, a plain list of 4-element frames or `{matrix}` back to the VAE encoder's `mu` latent (padded / cut to `max_seq_len` frames). Flat triplet token lists are rejected with `ValueError`
- `_tokens_from_matrix_buffer(matrix)` — `compound.matrix_to_frames`: piano-roll matrix → frames sorted by `(step, pitch)` plus EOS

The token format is described in §15 (compound frames).

---

### 3.3 RBFServerController

The main coordinator class. Constructed once at server startup.

**Initialization sequence:**
1. Selects the best available checkpoint kind
2. Instantiates `CudaRBFRuntime`, `RealtimeDecoderManager`, `FullFieldRenderManager`
3. Loads the latent pool from the dataset (N samples, encoded to `mu` vectors)
4. Places N anchor points on a circle (radius = 75% of field half-extent)
5. Precomputes initial RBF weights and BPM weights

**Key methods:**

| Method | Description |
|---|---|
| `process_frame(payload)` | Main hot path. Parses payload, updates state, interpolates, returns response |
| `_compute_latent_with_fallback(...)` | Dispatches to the active fallback mode for cursor latent |
| `_interpolate_bpm(...)` | Dispatches BPM interpolation, respecting fallback mode |
| `_compute_heatmap(...)` | Computes normalized heatmap grid (magnitude or confidence) |
| `_build_response(...)` | Assembles the full response dict sent to the client |
| `reload_model(mode, previous)` | Hot-swaps the VAE model, resets all state |
| `_load_latent_pool()` | Encodes the dataset to latent vectors, sets labels and BPMs |

**Signature caching:** RBF weights and heatmap are not recomputed on every frame. A tuple signature `(shape, rounded_bytes, sigma)` is stored; recomputation only happens when the signature changes.

---

### 3.4 FullFieldRenderManager

Runs a full `grid_resolution × grid_resolution` decode pass in a background daemon thread. Cells are decoded in blocks by `CudaRBFRuntime.decode_latents_to_matrices()` and every frame sequence is padded with PAD frames to `hparams.max_seq_len`. The export `outputs/<branch>/field_render_*.npz` holds `sequences` `[N, L, 4]` (`int16`) plus `token_format`, `field_names`, `field_vocab_sizes`, `n_specials`, `pad/sos/eos_token`, `center_timing`, `width`, `height`, `max_seq_len`, BPMs and the interpolation data (no `sequences_1d` and no `*_token` metadata anymore). A cancel event allows the render to be aborted mid-run.

Progress is published thread-safely via a generation counter (`_latest_generation` / `_last_consumed_generation`). The main loop polls `consume_latest_update()` on each frame and forwards progress to the client.

---

### 3.5 RealtimeDecoderManager

Runs a single autoregressive decode in a background daemon thread using the `TransformerVAE` decoder (`decode_step` + `Decoder.pick_frame`, same constraints as the batched path). Every generated frame is one finished note, so a partial piano-roll update is published after each note, enabling the UI to display decoding progress in real time. A new decode job cancels any running one.

---

## 4. RBF Interpolation and Latent Field

### 4.1 Radial Basis Functions

The latent field is defined by N anchor points placed in 2-D space, each associated with a latent vector (the VAE encoder `mu` for a dataset sample). Given a cursor position, the latent is interpolated using exact RBF interpolation with a Gaussian kernel:

$$\phi(r) = e^{-r^2 / (2\sigma^2)}$$

RBF weights are solved once per control-point / sigma change via `compute_rbf_weights` (from `modules/radial_basis_function.py`), then evaluated at each query point via `rbf_interpolate_exact`. BPM values at anchors are interpolated the same way.

`sigma` (kernel width) is user-controllable. Larger sigma → broader, smoother interpolation. Smaller sigma → anchors have more localized influence.

---

### 4.2 Latent Fallback Modes

In regions of low aggregate anchor influence (i.e. the cursor is far from all anchors), a fallback strategy determines what latent is returned. The mode is selected by the user from the UI dropdown.

| Mode | Behavior |
|---|---|
| `preserve_current_behavior` | Pure RBF extrapolation; no fallback blending |
| `mean_latent` | Blends toward the mean of all anchor latents |
| `nearest_anchor` | Blends toward a Gaussian-weighted mean of all anchors |
| `last_valid_latent` | Blends toward the last latent from a high-influence region |
| `user_defined_anchor0` | Anchor 0 is hidden from the UI; used as the fallback latent. RBF runs over anchors [1:]. Blend uses the standard thresholds (`_FALLBACK_BLEND_LOW=0.001`, `_FALLBACK_BLEND_HIGH=0.20`) |
| `user_defined_anchor0_refined` | Same hidden-anchor-0 topology, but the "active" endpoint is a normalized Gaussian-weighted mean of visible anchor latents (never zero-decays). Uses user-controlled thresholds (`refined_blend_low`, `refined_blend_high`). Includes anchor-hit snap: cursor within `1e-4` of a visible anchor returns that anchor's latent exactly |

**Hidden anchor-0 topology:** In both `user_defined_anchor0*` modes, the server stores N control points internally but sends only N-1 to the frontend (hiding index 0). Heatmap computation, influence display, and drag indices are all adjusted accordingly. The `latent_fallback_mode` key in all saved/sent data identifies this topology so readers can interpret the point arrays correctly.

---

### 4.3 Blend Math

All fallback modes use a smoothstep-based alpha to blend between the fallback vector and the RBF vector:

$$\alpha = \text{smoothstep}\!\left(\frac{\text{influence\_sum} - \text{low}}{\text{high} - \text{low}}\right)$$

$$\text{smoothstep}(t) = t^2(3 - 2t), \quad t \in [0, 1]$$

$$\text{output} = (1-\alpha)\cdot\text{fallback} + \alpha\cdot\text{rbf}$$

Two parameterizations exist:
- `_blend_factor_from_influence(influence_sum)` — uses the global constants `_FALLBACK_BLEND_LOW=0.001`, `_FALLBACK_BLEND_HIGH=0.20`
- `_blend_factor_custom(influence_sum, low, high)` — uses user-supplied thresholds (for the `refined` mode)

---

### 4.4 BPM Interpolation

BPM values are stored per anchor and interpolated in parallel with latents. Two modes:

- **`gaussian_blend`** — weighted mean: $\text{BPM} = \sum_i w_i \cdot \text{bpm}_i \;/\; \sum_i w_i$ where $w_i$ are raw Gaussian influence values
- **`exact_rbf`** — exact RBF interpolation using separately computed BPM weights

Both modes respect the hidden-anchor-0 topology and fallback blending identically to the latent path.

---

## 5. Heatmap Visualization

The heatmap is a `grid_resolution × grid_resolution` array of normalized floats in [0, 1], rendered as a background color overlay in the UI. Two modes:

- **Magnitude** — evaluates the RBF field at each grid point and takes the L2 norm of the resulting latent vector. Highlights regions of high absolute latent activity.
- **Confidence** — sums raw Gaussian influence from all anchors at each grid point. Highlights regions that are well-covered by anchor proximity.

Both are min-max normalized. The heatmap is cached with a signature `(point_shape, rounded_bytes, sigma, grid, mode, fallback_mode)` and only recomputed when one of these changes. In hidden-anchor-0 modes, the heatmap uses `points[1:]` and `weights_ex0` to exclude anchor-0 influence.

---

## 6. Decoding: Latent → MIDI Matrix

The decoder is a Transformer cross-attending to a latent memory block. Decoding is greedy and autoregressive over **compound frames** (one frame = one note, see §15):

- **SOS frame** `[1,1,1,1]` → start of sequence (decoder input only)
- **Note frame** `[step, pitch, velocity, deviation]` — absolute 16th position in the 2-bar window, drum row, quantized velocity, micro timing
- **EOS frame** `[2,2,2,2]` → end of sequence, afterwards **PAD** frames `[0,0,0,0]`

Four parallel heads predict the four fields from the same decoder state. `Decoder.pick_frame` chooses them in the order step → pitch → velocity / deviation with constraints (switchable via `constrained`): the step never falls, at the same step the pitch must rise (if no higher pitch is left the next step or EOS is taken). This keeps the order `(step, pitch)` and avoids duplicates that the independent heads could otherwise produce. The step head also decides the end (EOS).

The output is a 2-D matrix `(height × width*n_timing)` where each cell encodes velocity (0.0–1.0) with timing offset. A note of `n_timing // 2` steps wide is drawn from the onset column.

`RealtimeDecoderManager` publishes partial matrices after each note event, so the UI piano-roll updates column-by-column as decoding progresses.

Exported frames (`sequences` in Section 10) are the decoder's own frames up to EOS, padded with PAD frames to `max_seq_len` (`RETOKENIZE = False`). With `RETOKENIZE = True` each cell is converted to a matrix and back with `compound.matrix_to_frames` first.

---

## 7. Model Checkpoint Management

Checkpoints are stored under `models/<branch>/` with filenames matching:

```
model_<TIMESTAMP>.pt                  (kind: "none")
model_best_val_<TIMESTAMP>.pt         (kind: "best_val")
model_best_train_<TIMESTAMP>.pt       (kind: "best_train")
model_last_<TIMESTAMP>.pt             (kind: "last")
```

The UI exposes a dropdown for checkpoint kind and a `previous` index (0 = most recent). The `reload_model(mode, previous)` method hot-swaps the full model and resets all interpolation state without restarting the server.

A `model_catalog` dict is built at startup and after each reload, listing all available checkpoints per kind with their timestamps. This catalog is included in every response so the frontend can render the dropdowns reactively.

---

## 8. Communication Protocol (SocketIO)

### 8.1 Events

| Direction | Event | Description |
|---|---|---|
| Client → Server | `bootstrap` | Request initial state on connect |
| Client → Server | `frame` | Per-frame update with cursor/point/slider state |
| Client → Server | `select_model` | Hot-swap model checkpoint |
| Server → Client | `bootstrap` | Full initial state response |
| Server → Client | `frame` | Per-frame response with latent, heatmap, matrix, BPM |
| Server → Client | `error` | Error message string |

### 8.2 Payload Structure (client → server)

```json
{
  "cursor": [x, y],
  "draggable_points": [[x0,y0], [x1,y1], ...],
  "button_values": {
    "heatmapMode": "magnitude" | "confidence",
    "influenceMode": "raw" | "percentage",
    "bpmInterpMode": "gaussian_blend" | "exact_rbf",
    "latentFallbackMode": "<mode>",
    "encodeMidi": false,
    "decodeEnabled": false,
    "renderEnabled": false
  },
  "slider_values": {
    "gridResolution": 30,
    "sigma": 0.30,
    "refinedBlendLow": 0.0002,
    "refinedBlendHigh": 0.45
  },
  "midi_buffer": null | { "tokens": [...] } | { "matrix": [[...]] },
  "latent_vector": null | [...]
}
```

### 8.3 Response Structure (server → client)

```json
{
  "cursor": [x, y],
  "draggable_points": [[x0,y0], ...],
  "heatmap": [[...], ...],
  "latent_vector": [...],
  "decoded_matrix": [[...], ...],
  "influence_raw": [...],
  "influence_percentage": [...],
  "influence_values": [...],
  "button_values": { ... },
  "slider_values": {
    "gridResolution": 30,
    "sigma": 0.30,
    "refinedBlendLow": 0.0002,
    "refinedBlendHigh": 0.45
  },
  "labels": ["name (BPM: 120.00)", ...],
  "anchor_bpms": [...],
  "current_bpm": 120.0,
  "field_half_extent": 2.0,
  "n_timing": 4,
  "recon_cols": 64,
  "height": 9,
  "decode_status": "idle" | "decoding" | "done" | "cancelled",
  "decode_progress_step": -1,
  "render_status": "idle" | "rendering" | "done" | "cancelled" | "error",
  "render_completed": 0,
  "render_total": 900,
  "render_output_path": null | "outputs/branch/field_render_....npz",
  "render_error": null | "...",
  "model_options": { "modes": { ... } },
  "model_selection": { "mode": "none", "previous": 0, "timestamp": "..." }
}
```

`draggable_points`, `labels`, `anchor_bpms`, `influence_raw`, and `influence_percentage` all have N-1 elements in `user_defined_anchor0*` modes (anchor 0 is hidden from the frontend).

---

## 9. UI and Visualization (template.html)

The frontend is a single-page application served from `template.html`. All state is held in a JavaScript object `S` that mirrors `FrontendState`.

**Layout:**
- **Left panel (controls):** Anchor labels with influence indicators, hover name display, current BPM readout
- **Center (field canvas):** 2-D drag field with heatmap background and draggable anchor points
- **Right panel (controls):** All sliders and buttons; piano-roll matrix visualization

**Key UI elements:**

| Element | Description |
|---|---|
| Drag field | Canvas showing heatmap + anchor points. Click/drag to move anchors or cursor |
| Sigma slider | Controls RBF kernel width |
| Grid resolution slider | Controls heatmap/render grid density |
| Heat mode button | Toggle `Heat: Mag` / `Heat: Conf` |
| Influence mode button | Toggle `Influence: Raw` / `Influence: %` |
| BPM mode button | Toggle `BPM: Gaussian` / `BPM: Exact` |
| Latent fallback dropdown | Select fallback mode (see §4.2) |
| Blend Low / Blend High sliders | Visible only in `user_defined_anchor0_refined` mode; control blend thresholds |
| Decode button | Enable/disable live piano-roll decoding |
| Render button | Trigger / cancel full-field batch render |
| Checkpoint kind dropdown | Select model checkpoint kind |
| Checkpoint timestamp dropdown | Select specific checkpoint by timestamp |
| Piano-roll matrix | Live visualization of decoded MIDI pattern |

**Anchor display:** In `user_defined_anchor0*` modes, anchor 0 is hidden from all UI — no label, no drag handle, no influence bar. Display indices shown to the user are offset accordingly.

**Show/hide behavior:** The `#refinedBlendControls` div is shown only when the latent fallback dropdown is set to `user_defined_anchor0_refined`.

**Frame loop:** The frontend emits `"frame"` on every mouse move, button click, or slider change. Frames are queued (`queuedPayload`) and flushed only when no frame is already in-flight, preventing backlog.

---

## 10. Full-Field Render Output Format (.npz)

When a render completes, a compressed NumPy archive is saved to `outputs/<branch>/field_render_<timestamp>.npz`.

### Reading the file

```python
import numpy as np
data = np.load("outputs/branch/field_render_2026-04-23_14-30-00.npz")
```

### Keys

| Key | Shape / Type | Description |
|---|---|---|
| `sequences` | `(grid², max_seq_len, 4)` int16 | Decoded compound frames per cell in row-major order: `[step, pitch, velocity, deviation]` with offset `n_specials` (3), EOS frame `[2,2,2,2]`, then PAD frames. Decode with `modules.compound.frames_to_notes` / `frames_to_matrix` |
| `bpms_1d` | `(grid²,)` float32 | Interpolated BPM at each cell, row-major order |
| `control_points` | `(N, 2)` float32 | Full anchor positions at render time, including hidden anchor 0 if applicable |
| `sequence_length` | scalar int | Frames per cell (= `max_seq_len`, 128 by default) |
| `grid_resolution` | scalar int | Grid cells per axis |
| `traversal_order` | str | Always `"row_major"` (top-left to bottom-right, row by row) |
| `bpm_interp_mode` | str | `"gaussian_blend"` or `"exact_rbf"` |
| `latent_fallback_mode` | str | Active fallback mode at render time (see §4.2). Use this to interpret `control_points`: in `user_defined_anchor0*` modes, index 0 is the hidden fallback anchor |
| `sigma` | scalar float | RBF kernel width used during render |
| `field_half_extent` | scalar float | Half the side length of the field (coordinates span `[-half, +half]` on each axis) |
| `total_points` | scalar int | Total cells rendered (`grid_resolution²`) |
| `branch` | str | Git branch name at render time |
| `timestamp` | str | Render start time, `YYYY-MM-DD_HH-MM-SS` (Europe/Berlin timezone) |
| `token_format` | str | `"compound_v1"` |
| `field_names`, `field_vocab_sizes` | `(4,)` | `step, pitch, velocity, deviation` and `[35, 57, 35, 36]` |
| `n_specials`, `pad_token`, `sos_token`, `eos_token`, `center_timing` | scalar int | `3`, `0`, `1`, `2`, `16` (raw deviation of an on-grid note) |
| `width`, `height`, `n_pitch`, `n_vel`, `n_timing`, `max_seq_len`, `dataset_name` | scalar | Format and model metadata of the render |

Older exports (`sequences_1d`, `*_token` keys) were written in the flat triplet format and cannot be read with the compound codec.

### Reconstructing grid coordinates

```python
import numpy as np
data = np.load("field_render_....npz")
g = int(data["grid_resolution"])
h = float(data["field_half_extent"])
cell_size = (2.0 * h) / g
ys = -h + (np.arange(g) + 0.5) * cell_size   # row → y coordinate
xs = -h + (np.arange(g) + 0.5) * cell_size   # col → x coordinate
# Cell (row, col) maps to (xs[col], ys[row])
```

### Reconstructing sequences per cell

```python
from modules import compound
seqs = data["sequences"]       # shape (g*g, max_seq_len, 4)
bpms = data["bpms_1d"]          # shape (g*g,)
cell_idx = row * g + col        # row-major index
cell_seq = seqs[cell_idx]       # compound frames for that cell
cell_notes = compound.frames_to_notes(cell_seq)   # [{step, pitch, vel, shift, ...}, ...]
cell_bpm = bpms[cell_idx]
```

---

## 11. Threading Model

| Thread | Owned by | Purpose |
|---|---|---|
| Main (Flask-SocketIO event loop) | `create_app` | Handles all SocketIO events, calls `process_frame` |
| Render job | `FullFieldRenderManager` | Batch decode all grid cells; cancellable |
| Decode job | `RealtimeDecoderManager` | Autoregressive decode of cursor latent; cancellable |
| Decode broadcast loop | `create_app` (background task) | Polls decoder for partial updates and pushes to client |
| Render broadcast loop | `create_app` (background task) | Polls render manager for progress and pushes to client |

All shared state in `RBFServerController` is protected by `self._lock` (a `threading.Lock`). `FullFieldRenderManager` and `RealtimeDecoderManager` each use their own `threading.RLock` internally and expose only thread-safe `consume_latest_update()` / `request_*()` methods.

Race conditions are avoided with a job-ID pattern: each new job increments `_current_job_id`; `_publish_update` silently drops updates from stale jobs.

---

## 12. Startup and Configuration

The server is started by running `rbf_server_app.py` directly or via the `main()` function.

**Port selection:** By default, `AUTOMATIC_PORT_SELECTION = True`. CUDA machines use port `5006`; all others use `5009`. Set `AUTOMATIC_PORT_SELECTION = False` to enter a port interactively.

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `RBF_SERVER_HOST` | `0.0.0.0` | Bind address |
| `RBF_SERVER_PORT` | Auto-selected | Override port |
| `RBF_SAMPLE_COUNT` | `10` | Number of dataset samples to encode as anchors |
| `RBF_SAMPLE_SEED` | `42` | Random seed for reproducible sample selection |

**Branch-scoped paths:** Model checkpoints and render outputs are scoped to the current git branch via `get_current_branch()` / `safe_branch_name()`. This allows multiple experiment branches to coexist without file conflicts.

---

## 13. Training Tab

The page has two tabs: **Latent Field** (inference, unchanged) and **Training**. Training always runs on the server in a separate worker process (`src/train_worker.py`); the browser is only GUI and monitoring. Reloading the page or connecting later shows the running state including the history so far.

### 13.1 What can be set

| Group | Parameters |
|---|---|
| Data | `dataset_name`, `validation_split`, `n_samples` (0 = all), `batch_size`, `seed` |
| Optimization | `n_epochs`, `learning_rate`, `scheduler_start_lr`, `scheduler_end_lr`, `scheduler_exp_curve_factor`, `scheduler_warmup_ratio` (LR warm-up fraction, 0 = off), `weight_decay` |
| Beta / annealing | `free_bits`, `annealing_slope`, `annealing_list` (table of position/beta rows) |
| Checkpoints & early stopping | `last_checkpoint_interval` (write `model_last_*.pt` every N epochs), `early_stopping` (toggle), `early_stopping_patience`, `early_stopping_min_delta` |
| Model | `max_seq_len` (frames per sequence), `norm_first` (pre-norm toggle), `d_model`, `n_heads`, `n_layers`, `ff_dim`, `dropout`, `latent_dim`, `n_memory_tokens` |
| Options | `torch_compile` (default on when CUDA and a C compiler exist), event interval |

The compound token format (`field_sizes`, `field_vocab_sizes`, `field_loss_weights`, `token_format`) is not editable in the GUI; change it in `hparams.py`. `max_seq_len` and `norm_first` are architecture fields (locked when resuming, taken from the checkpoint); a dataset with another length is padded or cut to the model's `max_seq_len` when it is loaded. Early stopping only counts once the beta annealing has finished; see `docs/training.md` ("Speed, stability and stopping options"). The values shown are read fresh from `hparams.py`. Only compound datasets are offered; legacy (triplet) datasets are rejected with a field error, and resuming from a legacy checkpoint fails with a clear message (§15.4). The extra chart *Reconstruction error per field* shows the cross entropy of the step, pitch, velocity and deviation heads per epoch.

**Modes:** *New* starts from scratch. *Resume / retrain* loads a checkpoint (`last`, `best_val`, `best_train`, `none`) including optimizer and scheduler state. In resume mode only fields you changed override the checkpoint's values; the architecture fields are locked and come from the checkpoint. Extra options: keep beta constant, warm-start LR, end-LR factor.

### 13.2 Behaviour

- **Start** validates everything (`validate_train_config`): number ranges, `d_model % n_heads == 0`, annealing positions ascending in [0, 1], dataset and checkpoint exist. Errors are returned per field and shown next to the inputs; no process is started on error. A second start while a run is active is rejected.
- **Inference is paused** while training runs: running render/decode jobs are cancelled, the model is released (`gc` and `empty_cache`), `app_status.training` is `true`, and the Latent Field tab shows the yellow idle banner "Training läuft". Model/dataset switching is blocked.
- **Stop** asks the worker to finish the current batch. It then saves `model_last_*.pt` and `model_*.pt` and ends with `stopped = true`. A second click on the (now "Force kill") button kills the process; the last finished epoch checkpoint stays.
- **After end, stop or failure** the server runs `recheck_resources()`: model and dataset are loaded again and the new checkpoints appear in the checkpoint menus. A crash of the worker only sets state `failed` with a message.
- **Save to hparams.py** writes only the changed values of the `hyper_params` dict (whitelist above). Comments, formatting and other entries stay as they are. A backup `hparams.py.bak` is written first and the result is validated (parse, exec, value check) before replacing the file. Changing `d_model` changes `d_base`, so `latent_dim` and `ff_dim` follow when they are expressions of `d_base`; the response lists these derived values. Training itself never writes `hparams.py`. The running inference model is unaffected until the server is restarted.
- On server exit a running worker is terminated (`atexit`).

### 13.3 Socket events and HTTP

| Direction | Event | Payload |
|---|---|---|
| client → server | `train_get_config` | – |
| server → client | `train_config` | `defaults`, `datasets`, `model_options`, `options`, `branch`, `device`, `keys`, `state` |
| client → server | `train_start` | TrainConfig (below) |
| server → client | `train_start_result` | `{ok, errors: {field: message}}` |
| client → server | `train_stop` | `{force?: bool}` |
| server → client | `train_stop_result` | `{ok}` |
| client → server | `hparams_save` | `{changes: {key: value}}` |
| server → client | `hparams_save_result` | `{ok, changed, unchanged, derived, backup, message}` or `{ok: false, errors, message}` |
| server → client (broadcast, ≤ 4 Hz, on change) | `train_update` | snapshot: `state`, `active`, `stopped`, `config`, `epochs[]`, `batches[]`, `last_batch`, `logs[]`, `checkpoints[]`, `error`, `elapsed`, … |

`GET /api/train/status` returns the same snapshot. `GET /api/health` additionally contains `training`.

States: `idle | starting | running | stopping | finished | failed`.

```text
TrainConfig:
  mode: "new" | "resume"
  resume: {mode: best_val|best_train|last|none, previous: int, continue_beta: bool, start_lr?: float, end_lr_factor?: float}
  overrides: {key: value}          # whitelist keys only
  options: {torch_compile: bool, log_interval: int}
```

### 13.4 Worker event protocol

`train_worker.py --config <json> --stop-file <path>` writes one JSON object per stdout line (everything else, including library output, goes to stderr):

```text
{"type":"status","text":str}
{"type":"batch","epoch":int,"step":int,"steps":int,"loss":f,"beta":f,"lr":f,"step_time":f}
{"type":"epoch","epoch":int,"epochs":int,"train":{loss,rec,kl,ppl},"val":{...}|null,"beta":f,"lr":f,"epoch_time":f,"elapsed":f,"remaining":f}
{"type":"checkpoint","kind":"last|best_train|best_val|final","path":str}
{"type":"log","text":str}
{"type":"done","stopped":bool,"result":{...}}   |   {"type":"error","message":str}
```

The worker stops on SIGTERM/SIGINT or when the stop file exists (Windows). The server keeps ring buffers of the last epochs, batches and log lines for late-joining clients.

### 13.5 Manual GUI checklist

The frontend cannot be tested without a browser. After changes to `template.html`, check:

1. Tabs switch; the Latent Field tab behaves as before (drag, decode, render). The render heartbeat only runs while that tab is visible.
2. Training tab shows branch, device and dataset count; the form is filled from `hparams.py`.
3. Invalid input (e.g. `d_model = 30`, `n_heads = 4`, annealing positions descending, negative LR) shows errors next to the fields and starts nothing.
4. Start: banner "Training läuft", form locked, Stop enabled, progress bar, charts and log fill live; the Latent Field tab shows the yellow idle banner and its selectors are disabled.
5. Reload the page during training: state, charts and log are restored.
6. Stop: state `stopping`, then `finished (stopped)`; `model_last_*` is listed; the Latent Field tab works again and the new checkpoint is in the menu.
7. Resume: pick a checkpoint, architecture fields are locked, change epochs/LR, start; the epoch counter and optimizer state continue.
8. Save to hparams.py: message with the changed values; `hparams.py` keeps its comments and `hparams.py.bak` exists.
9. Start the server without model/dataset: it runs idle, the Training tab is still usable (it needs a dataset in `data/<branch>/` to start).


---

## 14. Dataset Tab

The third tab turns the former scripts `data.py` (MIDI folder → dataset), `data_merge.py` and `data_augment.py` into GUI functions. As everywhere in the app, the **browser only shows things**: all paths, all computation and all files live on the server (`src/modules/dataset_tools.py`). Datasets always go to `<project_root>/data/<git-branch>/<name>.npz`.

### 14.1 Functions

| Area | What it does |
|---|---|
| **Server folder browser** | *Browse server…* lists the folders of the **server's** file system (shortcuts: Home, `midifiles`, project root, `/Volumes`, `/`). A folder is chosen with *Select this folder*; the server then counts the MIDI files below it (numbered duplicates like `x.1.mid` are ignored, as in `extract`). |
| **Extract** | Runs `MidiProcessor.extract` in a background thread with all options of `data.py` (pitch mapping, drums only, filter duplicates, clip, Roland, remap, randomize pitches, skip/take, window jump, `max_seq_len` = frames per sequence incl. EOS (default from `hparams.py`; windows with more notes are cut, the count is logged and shown in the statistics), velocity noise, quantize strength/randomness). The MIDI files are processed **in parallel by a process pool** (`Workers`: 0 = automatic, i.e. CPU cores - 1 but at most 8; 1 = single process; jobs with fewer than 8 files always run in one process). Results are collected in file order, so the dataset is the same for every worker count. The random parts (velocity noise, quantize randomness) use one seed per file derived from the optional `Seed` (empty = random base seed); the seed that was used is stored in the dataset `options`. If the pool breaks down, the remaining files are processed in the server process. Progress (file x of n, sequences so far, skipped-file counters, ETA, log) is shown live and survives a page reload. *Cancel job* stops before the next file and writes nothing. **Speed:** the per-note numbers (step, window, microtiming) are computed per instrument with NumPy and the window lookup is linear (it used to be a quadratic search over all earlier notes). The parsed content of every MIDI file (time signatures, tempo, notes per instrument) is cached in `<project>/.cache/midi_parse/` (one small `.npz` per file, keyed by path, size and modification time; a changed, corrupt or outdated entry is simply parsed again). Parsing does not depend on the extraction options, so repeated runs with other mapping, jump or seed settings skip it. `RBF_MIDI_CACHE=0` disables the cache, `RBF_MIDI_CACHE_DIR` moves it; `.cache/` is git-ignored and can be deleted at any time. The result is bit-identical with and without cache (`tests/test_extract_cache.py`). On the 23 files in `midifiles/` (sequential): about 2.9 s before, 0.8 s with a cold and 0.2 s with a warm cache. |
| **Merge** | Concatenates two or more datasets of the branch (= `data_merge.py`), optionally removing duplicate sequences. Sequences of different length are padded; a missing BPM is kept missing. |
| **Augment** | Note removal, pitch-line removal, global velocity scaling and micro-shifts (= `data_augment.py`); keep percentage, scale and shift lists and an optional seed are settable. The BPM of the source is now preserved in the output metadata. |
| **Dataset list and statistics** | All datasets of the branch with sequence count, size, type and date. A selected dataset shows a summary, histograms (pitch usage with labels, velocity, microtiming, notes per sequence, BPM) and the meta info. |
| **Sequence viewer** | Slider / ◀ ▶ / random / arrow keys. A piano roll shows the notes of the sequence (rows = pitches that occur, colour = velocity, micro timing as sub-step offset) with a playhead during playback. |
| **MIDI playback** | *Connect MIDI* asks the browser for Web MIDI access and lists the output ports **of the computer running the browser**. Start/Stop plays the current sequence (optionally looping) on the chosen port and channel (default 10). The pitch mapping is *chromatic + 36* (like the old server playback) or the generic MIDI mapping; the tempo is the stored BPM unless overridden. *Browser synth* plays a simple preview without a MIDI device. Stop silences hanging notes (note-off for sounding notes + CC 123). |

Web MIDI needs Chrome/Edge/Opera and a secure context: open the server as `http://localhost:<port>` (or via https). Events are scheduled with a short look-ahead (100 ms) so that Stop is nearly immediate.

### 14.2 Dataset file format and meta info

`.npz` with the keys

- `data` — `int` array `(N, max_seq_len)` of tokens (unchanged, so training and the latent field keep working)
- `metadata` — one `(filename, window_index[, bpm])` per sequence (unchanged)
- `info` — **new**, optional: a JSON string with the dataset meta info. Datasets without it (older ones) are shown as type *legacy* and get an `info` key the first time their meta info is saved.

Contents of `info`:

| Key | Edited by | Description |
|---|---|---|
| `description`, `notes`, `tags`, `custom` | user (Dataset tab → *Save meta info*) | free description, free notes, tag list, custom key/value pairs |
| `schema`, `kind` (`extract`/`merge`/`augment`/`legacy`), `branch`, `created`, `modified` | automatic | |
| `source` | automatic | MIDI folder + file count, source datasets of a merge, or the augmented dataset |
| `options` | automatic | all extraction / merge / augmentation options of the run |
| `stats` | automatic | summary (sequences, unique sequences, source files, mean notes/tokens, BPM range) and extraction counters |

Saving meta info rewrites the file atomically (temp file + `os.replace`); the sequences and metadata are not changed. Output names must be plain file names (letters, digits, space, `.`, `_`, `-`); existing datasets are only replaced with *overwrite* enabled, and a job never overwrites its own source.

### 14.3 Socket events and HTTP

| Direction | Event | Payload |
|---|---|---|
| client → server | `ds_get_state` | – → `ds_state` `{branch, data_dir, datasets[], job, mappings[], hparams}` |
| client → server | `ds_browse` `{path, show_hidden}` | → `ds_browse_result` `{ok, path, parent, dirs[], midi_files[], midi_count, shortcuts[], error}` |
| client → server | `ds_scan` `{path}` | → `ds_scan_result` `{ok, midi_files, folders, skipped_numbered_duplicates}` |
| client → server | `ds_start` `{kind: extract\|merge\|augment, params}` | → `ds_start_result` `{ok, kind, errors{field: message}}` |
| client → server | `ds_cancel` | → `ds_cancel_result` `{ok}` |
| client → server | `ds_info` `{name}` | → `ds_info_result` `{ok, dataset{info, stats, shape, path, …}}` |
| client → server | `ds_sequence` `{name, index}` | → `ds_sequence_result` `{ok, sequence{notes[], bpm, filename, window, …}}` |
| client → server | `ds_save_info` `{name, patch}` | → `ds_save_info_result` `{ok, info}` |
| server → all | `ds_update` | job snapshot `{state, kind, phase, progress, done, total, message, stats, logs[], result, error, elapsed, eta}` (≈ 4 Hz, only on change) |
| server → all | `ds_list` | `{datasets[], state}` after a job ended or meta info was saved |

`GET /api/dataset/status` returns the job snapshot. Only one dataset job runs at a time; a second `ds_start` is rejected with a message. Job states: `idle`, `running`, `cancelling`, `finished`, `failed`, `cancelled`. A failing job (even a hard `exit()` inside the extractor) only ends the job, never the server.

### 14.4 Tests

- `python tests/test_dataset_tools.py` — folder browser, extraction with progress, statistics, sequence view, meta info, legacy datasets, merge, augmentation, cancel, failure (isolated branch folder `data/test-dataset-tools`, removed afterwards).
- `python tests/test_dataset_server.py` — the same through the Socket.IO test client, including the broadcast of progress.

### 14.5 Manual GUI checklist

1. The tab bar shows *Latent Field | Training | Dataset*; `#dataset` opens the tab directly. The Latent Field tab behaves as before.
2. *Browse server…* shows server folders; Up, shortcuts, typing a path + Enter and *show hidden folders* work; *Select this folder* fills the field and shows the MIDI file count.
3. Start an extraction with a name that already exists: the error is shown at the name field. With a new name the progress bar, counters and log update live; reloading the page keeps the job visible; *Cancel job* stops it and nothing is written.
4. After a finished job the banner names the new dataset, which is selected automatically; it appears in the list, in the merge list, in the augment source menu and (after a moment) in the Training tab's dataset menu.
5. Statistics: five charts are drawn, hovering a bar shows its value; resizing the window redraws them.
6. Meta info: edit description, tags, notes and custom fields, *Save meta info*, reload the page and select the dataset again — the values are still there.
7. Sequence viewer: slider, ◀ ▶, arrow keys and *Random* change the piano roll and the info line (file, window, BPM, notes, frames incl. EOS).
8. Playback: *Connect MIDI* (Chrome/Edge, `http://localhost`), choose a port, channel 10, *Start playback* — notes arrive on the port, the playhead runs, *Stop* silences everything; with *Browser synth* a click track is audible. Leaving the tab stops playback.
9. Start the server without any dataset: the tab shows "no dataset found"; extraction still works and creates the first dataset.

---

## 15. Compound Token Format

The whole pipeline (Dataset tab, training, Latent Field) uses **compound tokens** (`token_format = "compound_v1"`). One frame describes exactly one note; the former flat triplet stream (`pitch, velocity, timing` + `STEP` tokens, 256 tokens, vocabulary 123) is gone. All offsets, sorting and matrix mapping live in one place: `src/modules/compound.py`.

### 15.1 Frame layout

| Field (tensor order) | Raw value | Index in the frame | Vocabulary |
|---|---|---|---|
| `step` | 0–31 (absolute 16th in the 2-bar window) | raw + 3 | 35 |
| `pitch` | 0–53 (chromatic drum row) | raw + 3 | 57 |
| `velocity` | 0–31 | raw + 3 | 35 |
| `deviation` | 0–32 (center 16 = on the grid) | raw + 3 | 36 |

Every field has its own index space: `PAD = 0`, `SOS = 1`, `EOS = 2`; a special frame carries the id in all four fields. A sequence holds the note frames sorted by `(step, pitch)`, one EOS frame and PAD frames up to `max_seq_len` (128, set in `hparams.py`). Notes with the same `(step, pitch)` in a window are merged (loudest velocity wins, e.g. flams). Windows with more than `max_seq_len - 1` notes are cut after time; their number is shown in the extraction log and in the dataset statistics (*Truncated sequences*). SOS is only prepended as decoder input. The matrix mapping is unchanged: column = `step * n_timing + (deviation - center)`, note length `n_timing // 2`, wrapped at the window end.

### 15.2 Files

- Dataset `.npz`: `data` `(N, max_seq_len, 4)` int16, `metadata` `(file name, window, BPM)`, `info` JSON with `token_format`, `field_names`, `field_vocab_sizes`, `n_specials`, `max_seq_len`.
- Checkpoints: `hparams` contains `token_format`, `field_vocab_sizes`, `field_loss_weights`.
- Render export: see §10.

### 15.3 Model

- Input embedding: sum of four field embeddings (`FieldEmbedding`, scaled by `1/sqrt(4)`) plus the learned position embedding over the frame index; there is no beat embedding anymore (the position in the window is the step field).
- Encoder: padding mask from the step field, CLS pooling, `fc_mu` / `fc_logvar` as before.
- Decoder: one `Linear(d_model, 163)` split into the four heads (`field_slices`), KV-cache (`init_cache`, `decode_step(frame)`), `pick_frame`, `generate` → `[B, L, 4]`.
- Loss: sum of the per field cross entropies (weights `field_loss_weights`), averaged over non-PAD frames; on EOS frames only the step head counts. KL is normalised per frame. Training events and the CSV log contain the cross entropy per field; the Training tab shows it as an extra chart (solid = train, dashed = val) and in the status line.

### 15.4 Legacy data and models

- Datasets in the old 2D format appear in the Dataset tab as `legacy (triplet)`: they can be viewed (list, size), but statistics and piano roll are empty, and they cannot be merged, augmented, trained or used for inference. `load_dataset` rejects them with "legacy triplet dataset, re-extract it in the Dataset tab". The Training tab and the Latent Field only list compound datasets; if only legacy datasets exist the banner says so.
- Checkpoints without `token_format = "compound_v1"` are rejected by `load_model` before `load_state_dict` (`LegacyCheckpointError`); the server stays idle and shows the message in the banner. They still appear in the checkpoint menu.
- Encode MIDI accepts `{matrix}`, `{frames: [[s,p,v,d], ...]}` or a list of frames; flat token lists raise an error.

### 15.5 Not migrated (follow-up work)

The offline scripts still use the flat triplet format and do not run with compound models: `evaluation*.py`, `inference*.py`, `rbf.py`, `rbf_interpolation*.py`, `convert_to_onnx.py` / `ONNXWrapper`, `midi_mapping_visualizer*.py` and the older tests `tests/test_0*.py`, `tests/test.py`, `tests/onnx_test_*.py`, `tests/test_emptytokenlist.py`.

### 15.6 Tests

- `python tests/test_compound_codec.py` — codec, sorting, duplicates, truncation, matrix equality with the former triplet parser, matrix round trip.
- `python tests/test_compound_model.py` — field embedding, KV-cache vs. `forward` (post- and pre-norm), batch vs. single, `pick_frame` edge cases, loss masks, legacy checkpoint check, overfit on 8 sequences.
- `python tests/test_compound_pipeline.py` — idle start, extraction from `midifiles/`, mini training through the server, bootstrap ready, realtime decode, 4×4 render export, Encode MIDI.
- Adjusted: `tests/test_kv_cache.py`, `tests/test_dataset_tools.py`, `tests/test_dataset_server.py`, `tests/test_training_stop.py`, `tests/test_train_server.py`.

### 15.7 Manual checklist (compound)

1. Dataset tab: extract a MIDI folder; the summary shows *Token format compound_v1*, *Frames / sequence* and *Truncated sequences*; the piano roll and playback work.
2. An old dataset is listed in grey as `legacy (triplet)`, cannot be ticked in the merge list and is missing in the augment source menu and in the Training tab.
3. Training tab: start a run on the new dataset; the chart *Reconstruction error per field* and the status line `field CE: …` update per epoch.
4. Latent Field: with the new checkpoint the banner disappears; decoding fills the piano roll note by note; a full-field render writes `field_render_*.npz` with `sequences` `[N, 128, 4]`.
5. Select an old checkpoint: the banner shows the legacy message and the field stays idle; switching back to a compound checkpoint restores inference.


## 16. Reconstruction Viewer (Latent Field tab)

The Latent Field tab uses two columns of equal width (50:50). Long texts wrap, so the XY field keeps its size while a render runs.

The *Reconstruction* view is a piano roll in the style of *Dataset > Sequence viewer*:

- The server sends the decoded note matrix (`decoded_matrix`) plus `pitch_labels` and `generic_pitches` (chromatic index -> generic MIDI note). The browser recovers the notes with the same onset rule as `compound.matrix_to_notes`.
- **Pitch range** (default 24) sets the shown rows `0 .. range-1`. If notes above the range are generated, an orange warning appears above the canvas and a small `above range` marker in the plot. Those notes are still played.
- **Fold piano roll** shows only the used pitches inside the range.
- **MIDI playback** has the same controls as the Sequence viewer: output port (Web MIDI or browser synth), channel, pitch mapping, BPM override (empty = BPM interpolated at the cursor), loop, quantize (0-100 %, applied live) and Start/Stop. While playing, the notes follow the live reconstruction (cursor moves, decoding progress) and continue at the current position. Leaving the tab stops playback.
- **Show / Anchor point** choose what the piano roll and the playback use: `Reconstruction only` (default), `Original anchor only` (the notes of the dataset sequence the selected anchor was encoded from, orange) or `Anchor + reconstruction` (anchor as orange translucent outline behind the teal reconstruction; both are played). The anchor works without *Decode*. The list shows the anchors that are visible in the field (anchor 0 is hidden in the "Anchor 0 fallback" modes). Pitch range, fold, quantize and playback apply to the anchor notes as well.
- The server sends the original anchor notes as raw frames `[step, pitch, velocity, deviation]` (`anchor_notes`) together with `pool_version`, `n_vel`. They are only included when the client's `anchor_notes_version` (sent in every frame payload) differs from the current `pool_version`, i.e. after a dataset / sample change.
- The settings are stored in the browser (`localStorage`, keys `rc.*`).

Manual check: turn on *Decode*, move the cursor, change *Pitch range* (e.g. 12 to see the warning), start playback and move the quantize slider. Then set *Show* to `Original anchor only`, pick other anchors, switch to `Anchor + reconstruction` and start playback.

### 16.1 Real-time decoding, inference time, hold button, model info

- **Worker pool.** The real-time decoder (`RealtimeDecoderManager`) no longer cancels a running decode when the cursor enters a new cell. Up to `RBF_DECODE_WORKERS` (default 3) decodes run in parallel, each with its own CUDA stream. While all workers are busy only the newest request waits (older waiting requests are dropped, they were never started). A finished decode replaces the displayed reconstruction as a whole, and only if no newer result is shown already, so the display never jumps back to an older latent. Switching *Decode* off or changing the anchor pool discards queued and running results (`discard()`).
- **Settings (environment variables).** `RBF_DECODE_WORKERS` (parallel decodes, default 3), `RBF_DECODE_STREAM=1` (old live mode: the reconstruction grows note by note; a newer request then supersedes the older decode again), `RBF_DECODE_CHECK_EVERY` (default 8: the GPU -> CPU "all finished" test of the generation loop only runs every N steps; also used by the full-field render).
- **Inference time.** The server measures the time of every decode (without waiting time) and sends it as `decode_ms`. The info line of the reconstruction shows it next to the status, e.g. `decode: done (6.2 ms)`; while a new decode runs it shows `last ... ms`.
- **Hold: nearest anchor.** Next to the info line there is a button (only visible when *Show* is `Reconstruction only`). While it is held, the original notes of the anchor nearest to the cursor are laid over the reconstruction (orange outline) and played with it. Releasing it (or leaving the button / window) goes back to the reconstruction only.
- **Model parameters.** The *Model Checkpoint* section shows the most important parameters of the active model (`model_info`: parameter count, d_model, layers, heads, ff_dim, latent_dim, memory tokens, max_seq_len, pre/post-norm, dropout, format, dataset, device).

Manual check: turn on *Decode* and move the cursor quickly across cells (the reconstruction should keep updating smoothly instead of staying empty), hold *Hold: nearest anchor*, check the model parameters and the decode time. Test script: `tests/test_realtime_decoder.py`.
