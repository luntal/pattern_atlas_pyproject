import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, Slider

from modules.gui import CursorController, DraggablePoint
from modules.model import EncoderWrapper
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact
from modules.utilities import load_dataset, load_model, pick_reproducible_samples


class LatentFieldViewer:
    def __init__(
        self,
        latent_vectors: np.ndarray,
        labels: list[str],
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
        self.grid_resolution = 40
        self.heatmap_mode = 'magnitude'
        self.influence_mode = 'raw'
        self.control_points = self._make_initial_control_points(len(self.latent_vectors))
        self.weights = compute_rbf_weights(self.control_points, self.latent_vectors, sigma=self.sigma)

        self.fig = plt.figure(figsize=(8, 9))
        self.gs = self.fig.add_gridspec(18, 1, hspace=0.65)
        self.ax_field = self.fig.add_subplot(self.gs[:11, 0])
        self.ax_vector = self.fig.add_subplot(self.gs[12:14, 0])
        self.ax_influence = self.fig.add_subplot(self.gs[15:17, 0])

        self._setup_field_axis()
        self._setup_vector_axis()
        self._setup_influence_axis()
        self._setup_controls()
        self._connect_interaction()
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

    def _setup_controls(self):
        self.ax_resolution_slider = self.fig.add_axes([0.20, 0.06, 0.60, 0.025])
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

        self.ax_mode_button = self.fig.add_axes([0.20, 0.02, 0.12, 0.03])
        self.mode_button = Button(self.ax_mode_button, 'Mode: Mag')
        self.mode_button.label.set_fontsize(8)
        self.mode_button.on_clicked(self.on_mode_click)

        self.ax_influence_button = self.fig.add_axes([0.34, 0.02, 0.12, 0.03])
        self.influence_button = Button(self.ax_influence_button, 'Influence: Raw')
        self.influence_button.label.set_fontsize(8)
        self.influence_button.on_clicked(self.on_influence_mode_click)

        self.ax_sigma_slider = self.fig.add_axes([0.52, 0.02, 0.28, 0.03])
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
    viewer = LatentFieldViewer(mu, labels)
    viewer.show()


if __name__ == '__main__':
    main()