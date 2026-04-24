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

- `decode_latent_to_matrix(latent_vector)` — synchronous greedy autoregressive decoding of a latent into a piano-roll matrix `(height × width*n_timing)`
- `encode_midi_buffer_to_latent(midi_buffer)` — encodes a MIDI token sequence or matrix back to the VAE encoder's `mu` latent
- `_tokens_from_matrix_buffer(matrix)` — converts a piano-roll matrix back to the token vocabulary for re-encoding

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

Runs a full `grid_resolution × grid_resolution` decode pass in a background daemon thread. Each cell is decoded by `CudaRBFRuntime.decode_latent_to_matrix()` and its token sequence is collected into a flat array. A cancel event allows the render to be aborted mid-run.

Progress is published thread-safely via a generation counter (`_latest_generation` / `_last_consumed_generation`). The main loop polls `consume_latest_update()` on each frame and forwards progress to the client.

---

### 3.5 RealtimeDecoderManager

Runs a single autoregressive decode in a background daemon thread using the `TransformerVAE` decoder. Publishes partial piano-roll updates after each note event, enabling the UI to display decoding progress in real time. A new decode job cancels any running one.

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

The decoder is a Transformer cross-attending to a latent memory block. Decoding is greedy autoregressive with a custom token vocabulary:

- **SOS** → start of sequence
- **Pitch tokens** — one per drum track (row in the piano-roll)
- **Velocity tokens** — quantized velocity level
- **Timing tokens** — sub-step timing offset relative to the beat center
- **Step token** — advances the beat counter
- **EOS / PAD** — terminate / fill

The output is a 2-D matrix `(height × width*n_timing)` where each cell encodes velocity (0.0–1.0) with timing offset. A note of `n_timing // 2` steps wide is drawn from the onset column.

`RealtimeDecoderManager` publishes partial matrices after each note event, so the UI piano-roll updates column-by-column as decoding progresses.

Important nuance for exported token data (`sequences_1d` in Section 10): the saved sequence is not a direct byte-for-byte dump of the decoder's internal autoregressive token stream. For full-field export, each cell is decoded to a matrix first, then re-tokenized from that matrix, and finally normalized to a fixed length (`sequence_length`, currently 256) via truncate/pad. This matrix round-trip is intended to preserve project semantics, but token IDs can differ from a hypothetical raw decoder stream while remaining musically equivalent or near-equivalent.

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
| `sequences_1d` | `(grid² × sequence_length,)` int64 | All decoded token sequences concatenated in row-major order. Reshape to `(grid², sequence_length)` to get one sequence per cell |
| `bpms_1d` | `(grid²,)` float32 | Interpolated BPM at each cell, row-major order |
| `control_points` | `(N, 2)` float32 | Full anchor positions at render time, including hidden anchor 0 if applicable |
| `sequence_length` | scalar int | Fixed token sequence length per cell (always 256) |
| `grid_resolution` | scalar int | Grid cells per axis |
| `traversal_order` | str | Always `"row_major"` (top-left to bottom-right, row by row) |
| `bpm_interp_mode` | str | `"gaussian_blend"` or `"exact_rbf"` |
| `latent_fallback_mode` | str | Active fallback mode at render time (see §4.2). Use this to interpret `control_points`: in `user_defined_anchor0*` modes, index 0 is the hidden fallback anchor |
| `sigma` | scalar float | RBF kernel width used during render |
| `field_half_extent` | scalar float | Half the side length of the field (coordinates span `[-half, +half]` on each axis) |
| `total_points` | scalar int | Total cells rendered (`grid_resolution²`) |
| `branch` | str | Git branch name at render time |
| `timestamp` | str | Render start time, `YYYY-MM-DD_HH-MM-SS` (Europe/Berlin timezone) |

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
seqs = data["sequences_1d"].reshape(g * g, int(data["sequence_length"]))
bpms = data["bpms_1d"]          # shape (g*g,)
cell_idx = row * g + col        # row-major index
cell_seq = seqs[cell_idx]       # token sequence for that cell
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
