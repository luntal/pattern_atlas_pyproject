import os
import threading
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

from modules.gui import CursorController, DraggablePoint
from modules.model import EncoderWrapper
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact
from modules.utilities import load_dataset, load_model, pick_reproducible_samples


class RealtimeDecoderManager:
    def __init__(
        self,
        decoder: torch.nn.Module,
        hparams: SimpleNamespace,
        max_len: int,
        temperature: float = 0.1,
        sample: bool = False,
    ):
        self.decoder = decoder
        self.hparams = hparams
        self.max_len = int(max_len)
        self.temperature = float(temperature)
        self.sample = bool(sample)
        self.device = next(decoder.parameters()).device

        self._lock = threading.Lock()
        self._current_job_id = 0
        self._cancel_event = None
        self._thread = None

        cols = int(self.hparams.width * self.hparams.n_timing)
        self._latest_matrix = np.zeros((int(self.hparams.height), cols), dtype=np.float32)
        self._latest_progress_step = -1
        self._latest_status = 'idle'
        self._latest_generation = 0
        self._last_consumed_generation = -1

    def request_decode(self, latent_vector: np.ndarray, seed_matrix: np.ndarray):
        latent_vector = np.asarray(latent_vector, dtype=np.float32)
        seed_matrix = np.asarray(seed_matrix, dtype=np.float32)
        with self._lock:
            self._current_job_id += 1
            job_id = self._current_job_id
            if self._cancel_event is not None:
                self._cancel_event.set()
            cancel_event = threading.Event()
            self._cancel_event = cancel_event
            self._latest_status = 'decoding'

        thread = threading.Thread(
            target=self._decode_job,
            args=(job_id, latent_vector, seed_matrix.copy(), cancel_event),
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()

    def consume_latest_update(self):
        with self._lock:
            if self._latest_generation == self._last_consumed_generation:
                return None
            self._last_consumed_generation = self._latest_generation
            return {
                'matrix': self._latest_matrix.copy(),
                'progress_step': int(self._latest_progress_step),
                'status': self._latest_status,
            }

    def shutdown(self):
        with self._lock:
            if self._cancel_event is not None:
                self._cancel_event.set()

    def _publish_update(self, job_id: int, matrix: np.ndarray, progress_step: int, status: str):
        with self._lock:
            if job_id != self._current_job_id:
                return
            self._latest_matrix = matrix.astype(np.float32, copy=True)
            self._latest_progress_step = int(progress_step)
            self._latest_status = status
            self._latest_generation += 1

    def _write_wrapped_segment(self, mat: np.ndarray, row: int, start: int, length: int, value: float):
        cols = mat.shape[1]
        if length <= 0:
            return
        length = min(length, cols)
        s = start % cols
        e = s + length
        if e <= cols:
            mat[row, s:e] = value
        else:
            first_len = cols - s
            mat[row, s:cols] = value
            mat[row, 0:(length - first_len)] = value

    def _decode_job(self, job_id: int, latent_vector: np.ndarray, seed_matrix: np.ndarray, cancel_event: threading.Event):
        h = self.hparams
        n_timing = int(h.n_timing)
        cols = int(h.width * n_timing)
        min_pitch_token = int(h.min_pitch_token)
        max_pitch_token = int(h.max_pitch_token)
        min_vel_token = int(h.min_vel_token)
        max_vel_token = int(h.max_vel_token)
        min_timing_token = int(h.min_timing_token)
        max_timing_token = int(h.max_timing_token)
        center_timing_token = int(h.center_timing_token)
        step_token = int(h.step_token)
        eos_token = int(h.eos_token)
        sos_token = int(h.sos_token)

        partial_matrix = np.zeros_like(seed_matrix, dtype=np.float32)
        display_matrix = seed_matrix.copy()

        z = torch.from_numpy(latent_vector).to(self.device, dtype=torch.float32).unsqueeze(0)
        memory = self.decoder.z_to_mem(z).view(1, self.decoder.n_memory_tokens, self.decoder.emb_size)
        cur_tokens = torch.full((1, 1), sos_token, dtype=torch.long, device=self.device)
        beat_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        step_idx = 0
        pending_pitch = None
        pending_vel = None

        for _ in range(self.max_len):
            if cancel_event.is_set():
                self._publish_update(job_id, display_matrix, step_idx, 'cancelled')
                return

            T = cur_tokens.size(1)
            if T > self.decoder.max_seq_len:
                break

            with torch.no_grad():
                emb = self.decoder.token_embedding(cur_tokens)
                beat_emb = self.decoder.beat_embedding(beat_tokens)
                emb = self.decoder.projection(torch.cat((emb, beat_emb), dim=-1))
                pos_emb = self.decoder.pos_embedding(torch.arange(T, device=self.device)).unsqueeze(0)
                emb = emb + pos_emb
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(self.device)
                tgt_key_padding_mask = (cur_tokens == self.decoder.pad_token).float()
                out = self.decoder.transformer_decoder(
                    tgt=emb,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                logits = self.decoder.fc_out(out)
                last_logits = logits[:, -1, :] / max(1e-8, self.temperature)
                probs = torch.nn.functional.softmax(last_logits, dim=-1)
                if self.sample:
                    next_tok = torch.multinomial(probs, num_samples=1)
                else:
                    next_tok = torch.argmax(probs, dim=-1, keepdim=True)

            token = int(next_tok.item())
            cur_tokens = torch.cat([cur_tokens, next_tok], dim=1)
            is_step = (cur_tokens == step_token).to(torch.long)
            beat_tokens = torch.cumsum(is_step, dim=1)
            beat_tokens += 1
            beat_tokens[:, 0] = 0

            if token == eos_token:
                break
            if token == step_token:
                step_idx += 1
                continue

            if min_pitch_token <= token < max_pitch_token:
                pending_pitch = token
                pending_vel = None
                continue
            if pending_pitch is not None and min_vel_token <= token < max_vel_token:
                pending_vel = token
                continue
            if pending_pitch is not None and pending_vel is not None and min_timing_token <= token < max_timing_token:
                pitch_idx = pending_pitch - min_pitch_token
                vel = (pending_vel - min_vel_token) / max(1, (h.n_vel - 1))
                timing = token - center_timing_token

                start = step_idx * n_timing + timing
                note_len = max(1, n_timing // 2)
                self._write_wrapped_segment(partial_matrix, pitch_idx, start, note_len, float(vel))

                frontier = max(0, min(cols, step_idx * n_timing + note_len))
                display_matrix[:, :frontier] = partial_matrix[:, :frontier]
                self._publish_update(job_id, display_matrix, step_idx, 'decoding')

                pending_pitch = None
                pending_vel = None

        display_matrix[:, :] = partial_matrix
        self._publish_update(job_id, display_matrix, step_idx, 'done')


class LatentFieldViewer:
    def __init__(
        self,
        latent_vectors: np.ndarray,
        labels: list[str],
        decoder: torch.nn.Module,
        hparams: SimpleNamespace,
        field_size: float = 4.0,
        sigma: float = 0.3,
        vector_range: tuple[float, float] = (-4.0, 4.0),
    ):
        self.latent_vectors = np.asarray(latent_vectors, dtype=np.float32)
        self.labels = labels
        self.sigma = float(sigma)
        self.vector_range = vector_range
        self.field_half_extent = field_size / 2.0
        self.cursor_pos = np.zeros(2, dtype=np.float32)
        self.hparams = hparams
        self.grid_resolution = 40
        self.heatmap_mode = 'magnitude'
        self.influence_mode = 'raw'
        self.recon_cols = int(self.hparams.width * self.hparams.n_timing)
        self.reconstruction_matrix = np.zeros((int(self.hparams.height), self.recon_cols), dtype=np.float32)
        self.control_points = self._make_initial_control_points(len(self.latent_vectors))
        self.weights = compute_rbf_weights(self.control_points, self.latent_vectors, sigma=self.sigma)
        self.decoder_manager = RealtimeDecoderManager(
            decoder=decoder,
            hparams=hparams,
            max_len=int(self.hparams.max_seq_len),
            temperature=0.1,
            sample=False,
        )

        self.fig = plt.figure(figsize=(9, 10))
        self.gs = self.fig.add_gridspec(21, 1, hspace=0.80)
        self.ax_field = self.fig.add_subplot(self.gs[:10, 0])
        self.ax_vector = self.fig.add_subplot(self.gs[11:13, 0])
        self.ax_influence = self.fig.add_subplot(self.gs[14:16, 0])
        self.ax_reconstruction = self.fig.add_subplot(self.gs[18:21, 0])

        self._setup_field_axis()
        self._setup_vector_axis()
        self._setup_influence_axis()
        self._setup_reconstruction_axis()
        self._setup_controls()
        self._connect_interaction()

        self.recon_timer = self.fig.canvas.new_timer(interval=50)
        self.recon_timer.add_callback(self._poll_decoder_updates)
        self.recon_timer.start()
        self.fig.canvas.mpl_connect('close_event', self._on_close)

        self.update_display()

    def _make_initial_control_points(self, n_points: int) -> np.ndarray:
        if n_points == 0:
            return np.zeros((0, 2), dtype=np.float32)
        radius = self.field_half_extent * 0.75
        angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        x = np.cos(angles) * radius
        y = np.sin(angles) * radius
        return np.column_stack((x, y)).astype(np.float32)

    def _setup_field_axis(self):
        self.ax_field.set_title('RBF Latent Field')
        self.ax_field.set_xlim(-self.field_half_extent, self.field_half_extent)
        self.ax_field.set_ylim(-self.field_half_extent, self.field_half_extent)
        self.ax_field.set_aspect('equal', adjustable='box')
        self._update_field_ticks()
        self.ax_field.grid(True, which='minor', color='0.9', linewidth=0.45)
        self.ax_field.set_xlabel('x')
        self.ax_field.set_ylabel('y')

        self.heatmap_image = self.ax_field.imshow(
            np.zeros((self.grid_resolution, self.grid_resolution), dtype=np.float32),
            extent=[-self.field_half_extent, self.field_half_extent, -self.field_half_extent, self.field_half_extent],
            origin='lower',
            cmap='gray',
            alpha=0.25,
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )
        self.scatter_points = self.ax_field.scatter(
            self.control_points[:, 0], self.control_points[:, 1], color='tab:blue', s=45, zorder=3
        )
        self.scatter_cursor = self.ax_field.scatter(
            self.cursor_pos[0], self.cursor_pos[1], color='tab:red', s=110, marker='x', zorder=4
        )
        self.point_labels = [
            self.ax_field.text(point[0] + 0.05, point[1] + 0.05, f'[{i}] {label}', fontsize=7, visible=False)
            for i, (point, label) in enumerate(zip(self.control_points, self.labels))
        ]
        self.status_text = self.ax_field.text(
            0.02,
            0.98,
            '',
            transform=self.ax_field.transAxes,
            verticalalignment='top',
            fontsize=8,
            family='monospace',
        )

    def _setup_vector_axis(self):
        latent_dim = self.latent_vectors.shape[1]
        self.ax_vector.set_title('Interpolated Latent Vector')
        self.vector_image = self.ax_vector.imshow(
            np.zeros((1, latent_dim), dtype=np.float32),
            aspect='auto',
            cmap='coolwarm',
            interpolation='nearest',
            vmin=self.vector_range[0],
            vmax=self.vector_range[1],
        )
        self.ax_vector.set_yticks([])
        # self.ax_vector.set_xlabel('Latent dimension')
        self.fig.colorbar(self.vector_image, ax=self.ax_vector, orientation='horizontal', pad=0.5)

    def _setup_influence_axis(self):
        n_points = len(self.control_points)
        x_positions = np.arange(n_points)
        self.ax_influence.set_title('Anchor Influence')
        self.ax_influence.set_xlabel('Anchor index')
        self.ax_influence.set_ylabel('Raw influence')
        self.ax_influence.set_ylim(0.0, 1.0)
        if n_points > 0:
            self.influence_bars = self.ax_influence.bar(
                x_positions,
                np.zeros(n_points, dtype=np.float32),
                color='tab:green',
                alpha=0.75,
                width=0.8,
            )
            self.ax_influence.set_xlim(-0.5, n_points - 0.5)
            self.ax_influence.set_xticks(x_positions)
        else:
            self.influence_bars = []
            self.ax_influence.set_xlim(-0.5, 0.5)
            self.ax_influence.set_xticks([])

    def _setup_reconstruction_axis(self):
        n_timing = int(self.hparams.n_timing)
        self.ax_reconstruction.set_title('Realtime Decoded Reconstruction (32 sixteenth steps x 24 pitches)')
        self.reconstruction_image = self.ax_reconstruction.imshow(
            self.reconstruction_matrix,
            cmap='gray_r',
            interpolation='nearest',
            aspect='auto',
            vmin=0.0,
            vmax=1.0,
        )
        self.ax_reconstruction.invert_yaxis()
        self.ax_reconstruction.set_ylabel('Pitch index')
        self.ax_reconstruction.set_xlabel('Sixteenth-step grid')
        self.ax_reconstruction.set_xticks(np.arange(-0.5, self.recon_cols, n_timing), minor=True)
        self.ax_reconstruction.set_xticks(np.arange(0, self.recon_cols, 4 * n_timing))
        self.ax_reconstruction.set_xticklabels(np.arange(0, self.recon_cols, 4 * n_timing) // n_timing + 1)
        self.ax_reconstruction.set_yticks(np.arange(-0.5, self.hparams.height, 1), minor=True)
        self.ax_reconstruction.grid(which='minor', color='0.75', linestyle='-', linewidth=0.45)
        self.ax_reconstruction.tick_params(axis='both', which='major', labelsize=8)

    def _setup_controls(self):
        self.ax_resolution_slider = self.fig.add_axes([0.20, 0.05, 0.60, 0.02])
        self.resolution_slider = Slider(
            self.ax_resolution_slider,
            'Grid Res',
            4,
            60,
            valinit=self.grid_resolution,
            valstep=1,
        )
        self.resolution_slider.label.set_fontsize(8)
        self.resolution_slider.valtext.set_fontsize(8)
        self.resolution_slider.on_changed(self.on_resolution_change)

        self.ax_mode_button = self.fig.add_axes([0.20, 0.015, 0.12, 0.028])
        self.mode_button = Button(self.ax_mode_button, 'Mode: Mag')
        self.mode_button.label.set_fontsize(8)
        self.mode_button.on_clicked(self.on_mode_click)

        self.ax_influence_button = self.fig.add_axes([0.34, 0.015, 0.14, 0.028])
        self.influence_button = Button(self.ax_influence_button, 'Influence: Raw')
        self.influence_button.label.set_fontsize(8)
        self.influence_button.on_clicked(self.on_influence_mode_click)

        self.ax_sigma_slider = self.fig.add_axes([0.54, 0.015, 0.26, 0.028])
        self.sigma_slider = Slider(
            self.ax_sigma_slider,
            'Sigma',
            0.05,
            3.0,
            valinit=self.sigma,
            valstep=0.01,
        )
        self.sigma_slider.label.set_fontsize(8)
        self.sigma_slider.valtext.set_fontsize(8)
        self.sigma_slider.on_changed(self.on_sigma_change)

    def _connect_interaction(self):
        self.draggables = []
        for index in range(len(self.control_points)):
            draggable = DraggablePoint(
                self.control_points[index],
                index,
                self.scatter_points,
                self.update_display,
                self.fig,
                self.ax_field,
                self.control_points,
            )
            draggable.connect()
            self.draggables.append(draggable)

        self.cursor_controller = CursorController(
            self.cursor_pos,
            self.scatter_cursor,
            self.update_display,
            self.fig,
            self.ax_field,
            self.control_points,
        )
        self.cursor_controller.connect()
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('figure_leave_event', self._on_figure_leave)

    def _interpolated_latent(self) -> np.ndarray:
        return rbf_interpolate_exact(self.control_points, self.weights, self.cursor_pos, sigma=self.sigma)

    def _refresh_point_labels(self):
        for label_artist, point in zip(self.point_labels, self.control_points):
            label_artist.set_position((point[0] + 0.05, point[1] + 0.05))

    def _set_hovered_label(self, hovered_index: int | None) -> bool:
        changed = False
        for i, label_artist in enumerate(self.point_labels):
            should_show = i == hovered_index
            if label_artist.get_visible() != should_show:
                label_artist.set_visible(should_show)
                changed = True
        return changed

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax_field:
            if self._set_hovered_label(None):
                self.fig.canvas.draw_idle()
            return

        contains, info = self.scatter_points.contains(event)
        hovered_index = int(info['ind'][0]) if contains and len(info.get('ind', [])) > 0 else None
        if self._set_hovered_label(hovered_index):
            self.fig.canvas.draw_idle()

    def _on_figure_leave(self, event):
        if self._set_hovered_label(None):
            self.fig.canvas.draw_idle()

    def _on_close(self, event):
        self.decoder_manager.shutdown()

    def _request_realtime_decode(self, current_latent: np.ndarray):
        self.decoder_manager.request_decode(current_latent.astype(np.float32), self.reconstruction_matrix)

    def _poll_decoder_updates(self):
        update = self.decoder_manager.consume_latest_update()
        if update is None:
            return

        self.reconstruction_matrix = update['matrix']
        self.reconstruction_image.set_data(self.reconstruction_matrix)
        self.ax_reconstruction.set_title(
            'Realtime Decoded Reconstruction '
            f"(status={update['status']}, step={update['progress_step']})"
        )
        self.fig.canvas.draw_idle()

    def _update_field_ticks(self):
        # Draw only minor-grid lines based on current heatmap resolution.
        grid_ticks = np.linspace(-self.field_half_extent, self.field_half_extent, self.grid_resolution + 1)
        self.ax_field.set_xticks([])
        self.ax_field.set_yticks([])
        self.ax_field.set_xticks(grid_ticks, minor=True)
        self.ax_field.set_yticks(grid_ticks, minor=True)
        self.ax_field.tick_params(axis='both', which='major', length=0, labelbottom=False, labelleft=False)
        self.ax_field.tick_params(axis='both', which='minor', length=0)

    def _compute_anchor_influence_raw(self) -> np.ndarray:
        dists = np.linalg.norm(self.control_points - self.cursor_pos, axis=1)
        return np.exp(-(dists ** 2) / (2 * self.sigma ** 2))

    def _update_influence_plot(self):
        if len(self.influence_bars) == 0:
            return

        raw_influence = self._compute_anchor_influence_raw().astype(np.float32)
        if self.influence_mode == 'percentage':
            denom = float(np.sum(raw_influence))
            influence_values = raw_influence / denom if denom > 0.0 else np.zeros_like(raw_influence)
            self.ax_influence.set_ylabel('%')
            self.ax_influence.set_ylim(0.0, 1.0)
        else:
            influence_values = raw_influence
            self.ax_influence.set_ylabel('Raw')
            self.ax_influence.set_ylim(0.0, 1.0)

        for bar, value in zip(self.influence_bars, influence_values):
            bar.set_height(float(value))

    def _compute_heatmap(self) -> np.ndarray:
        """Compute heatmap based on current grid resolution and mode."""
        resolution = self.grid_resolution
        x = np.linspace(-self.field_half_extent, self.field_half_extent, resolution)
        y = np.linspace(-self.field_half_extent, self.field_half_extent, resolution)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)

        if self.heatmap_mode == 'magnitude':
            # Compute L2 norm of interpolated latent at each grid point
            heatmap = np.zeros(len(grid_points), dtype=np.float32)
            for i, point in enumerate(grid_points):
                latent = rbf_interpolate_exact(self.control_points, self.weights, point, sigma=self.sigma)
                heatmap[i] = np.linalg.norm(latent)
        else:  # 'confidence'
            # Compute RBF basis sum (interpolation weight/confidence at each point)
            distances = np.linalg.norm(
                grid_points[:, np.newaxis, :] - self.control_points[np.newaxis, :, :],
                axis=2,
            )
            rbf_basis = np.exp(-(distances ** 2) / (2 * self.sigma ** 2))
            heatmap = np.sum(rbf_basis, axis=1)

        return heatmap.reshape(resolution, resolution)

    def on_resolution_change(self, value):
        self.grid_resolution = int(value)
        self._update_field_ticks()
        self.update_display()

    def on_mode_click(self, event):
        self.heatmap_mode = 'confidence' if self.heatmap_mode == 'magnitude' else 'magnitude'
        mode_text = 'Mag' if self.heatmap_mode == 'magnitude' else 'Conf'
        self.mode_button.label.set_text(f'Mode: {mode_text}')
        self.update_display()

    def on_influence_mode_click(self, event):
        self.influence_mode = 'percentage' if self.influence_mode == 'raw' else 'raw'
        button_text = 'Influence: %' if self.influence_mode == 'percentage' else 'Influence: Raw'
        self.influence_button.label.set_text(button_text)
        self.update_display(isCursor=True)

    def on_sigma_change(self, value):
        self.sigma = float(value)
        self.weights = compute_rbf_weights(self.control_points, self.latent_vectors, sigma=self.sigma)
        self.update_display(isCursor=True)

    def update_display(self, isCursor: bool = False):
        if not isCursor:
            self.weights = compute_rbf_weights(self.control_points, self.latent_vectors, sigma=self.sigma)

        heatmap_data = self._compute_heatmap()
        heat_min = float(np.min(heatmap_data))
        heat_max = float(np.max(heatmap_data))
        if heat_max > heat_min:
            normalized_heatmap = (heatmap_data - heat_min) / (heat_max - heat_min)
        else:
            normalized_heatmap = np.zeros_like(heatmap_data, dtype=np.float32)

        self.heatmap_image.set_clim(0.0, 1.0)
        self.heatmap_image.set_alpha(0.22 if self.heatmap_mode == 'magnitude' else 0.28)
        self.heatmap_image.set_data(normalized_heatmap)

        current_latent = self._interpolated_latent().astype(np.float32)
        self._request_realtime_decode(current_latent)
        self.scatter_points.set_offsets(self.control_points)
        self.scatter_cursor.set_offsets([self.cursor_pos])
        self._refresh_point_labels()
        self._update_influence_plot()
        self.vector_image.set_data(current_latent.reshape(1, -1))
        self.status_text.set_text(
            f'cursor=({self.cursor_pos[0]:+.2f}, {self.cursor_pos[1]:+.2f})\n'
            f'mode={self.heatmap_mode[:4]} | grid={self.grid_resolution}x{self.grid_resolution} | sigma={self.sigma:.2f}\n'
            f'latent min={current_latent.min():+.2f} max={current_latent.max():+.2f}'
        )
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main():
    model, hparams = load_model(previous=0, onnx=False)
    data, metadata = load_dataset(hparams)
    data, metadata, _ = pick_reproducible_samples(data, metadata, sample_count=10, sample_seed=42)
    labels = [os.path.basename(meta[0]) for meta in metadata]

    encoder = EncoderWrapper(model.encoder, hparams)
    input_tokens = torch.from_numpy(data).long()
    with torch.no_grad():
        mu = encoder(input_tokens)

    print('mu shape: ', mu.shape)
    viewer = LatentFieldViewer(mu, labels, model.decoder, hparams)
    viewer.show()


if __name__ == '__main__':
    main()