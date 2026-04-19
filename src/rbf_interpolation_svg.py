import numpy as np
import matplotlib.pyplot as plt
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize
from modules.midi_processor import MidiProcessor
import torch
from modules.utilities import seq2mat, load_model, load_dataset, pick_reproducible_samples, create_latent_vectors
from matplotlib.widgets import Button, Slider
import time
import mido
import threading
from types import SimpleNamespace
import matplotlib.image as mpimg
import xml.etree.ElementTree as ET
import re
import os
import os.path
import pickle
import glob
from modules.radial_basis_function import compute_rbf_weights, rbf_interpolate_exact
from modules.gui import DraggablePoint, CursorController
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

"""
calculating RBF (Radial Basis Function) interpolation between multiple latent vectors

"""

##########################
# Load model with metadata
##########################

model, hparams = load_model()

########################################
# CONFIGURATION
########################################
def get_country_positions(svg_path, target_xlim, target_ylim, original_width=2000, original_height=857):
    country_bounds = {}

    try:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        for path in root.findall(".//{http://www.w3.org/2000/svg}path") + root.findall(".//path"):
            name = path.get('name') or path.get('class') or path.get('id')
            d_attr = path.get('d')

            if d_attr and name:
                # Tokenize: find all commands (letters) and numbers
                tokens = re.findall(r"([a-zA-Z])|([-+]?\d*\.\d+|\d+)", d_attr)

                points = []
                cur_x, cur_y = 0, 0
                cmd = ''

                for char, num in tokens:
                    if char:
                        cmd = char
                    elif num:
                        val = float(num)
                        # We process numbers in pairs
                        if not points or len(points[-1]) == 2:
                            points.append([val])
                        else:
                            points[-1].append(val)

                            # Apply the command logic to the newly completed point
                            px, py = points[-1]
                            if cmd in 'ml':  # Relative
                                cur_x += px
                                cur_y += py
                            elif cmd in 'ML':  # Absolute
                                cur_x = px
                                cur_y = py
                            elif cmd == 'h':
                                cur_x += px; points[-1] = [cur_x, cur_y]  # Relative Horizontal
                            elif cmd == 'H':
                                cur_x = px;  points[-1] = [cur_x, cur_y]  # Absolute Horizontal
                            elif cmd == 'v':
                                cur_y += px; points[-1] = [cur_x, cur_y]  # Relative Vertical (px is the single value)
                            elif cmd == 'V':
                                cur_y = px;  points[-1] = [cur_x, cur_y]  # Absolute Vertical

                            points[-1] = [cur_x, cur_y]

                if points:
                    coords = np.array(points)
                    p_min_x, p_min_y = np.min(coords, axis=0)
                    p_max_x, p_max_y = np.max(coords, axis=0)

                    if name not in country_bounds:
                        country_bounds[name] = [p_min_x, p_min_y, p_max_x, p_max_y]
                    else:
                        curr = country_bounds[name]
                        country_bounds[name] = [
                            min(curr[0], p_min_x), min(curr[1], p_min_y),
                            max(curr[2], p_max_x), max(curr[3], p_max_y)
                        ]

        positions = {}
        for name, bounds in country_bounds.items():
            mid_x = (bounds[0] + bounds[2]) / 2
            mid_y = (bounds[1] + bounds[3]) / 2
            mapped_x = ((mid_x / original_width) * (target_xlim[1] - target_xlim[0])) + target_xlim[0]
            mapped_y = ((1 - (mid_y / original_height)) * (target_ylim[1] - target_ylim[0])) + target_ylim[0]
            positions[name] = (mapped_x, mapped_y)

    except Exception as e:
        print(f"Error parsing SVG paths: {e}")
    return positions
scale = 2
ratio = 0.4
radius_x = 4
radius_y = radius_x * ratio
# Calculate the aspect ratio based on 2000x857
width_range = radius_x * 2  # -1.5 to 1.5
img_aspect = 857 / 2000
height_range = width_range * img_aspect
y_limit = height_range / 2
# Extract positions from the original world.svg
country_coords = get_country_positions("assets/world.svg", (-radius_x, radius_x), (-y_limit, y_limit))


country_list = list(country_coords.keys())
selected_countries = ['United States', 'Spain',  'New Zealand', 'China', 'Australia', 'Togo', 'Namibia', 'Chile', 'Germany']

# NUM_POINTS = len(selected_countries)  # number of control points / latent vectors
LATENT_DIM = hparams.latent_dim  # dimension of latent space
RBF_SIGMA = 0.3  # width of Gaussian RBF
grid_resolution = 4
last_recalculated_grid_resolution = grid_resolution
outputs_dir = 'outputs'
precomputed_reconstructions_path = os.path.join(outputs_dir, 'precomputed_reconstructions.pkl')
precomputed_latents_path = os.path.join(outputs_dir, 'precomputed_latents.pkl')
precomputed_sequences_path = os.path.join(outputs_dir, 'precomputed_sequences.pkl')
control_points_path = os.path.join(outputs_dir, 'control_points.npy')
recalc_thread = None
recalc_stop_event = None
recalc_lock = threading.Lock()
recalc_status = 'idle'
recalc_result = None
recalc_error = None
recalc_batch_size = 32

# Global buffer to store precomputed values
precomputed_reconstructions = None
precomputed_latents = None
l_min, l_max = None, None

midi_processor = MidiProcessor(hparams)


def infer_grid_resolution(n_items):
    if n_items is None:
        return None
    resolution = int(round(np.sqrt(n_items)))
    if resolution * resolution == n_items:
        return resolution
    return None


def load_cached_state():
    global precomputed_reconstructions, precomputed_latents, l_min, l_max
    global grid_resolution, last_recalculated_grid_resolution, nearest_x, nearest_y, weights

    loaded_any = False

    if os.path.exists(precomputed_reconstructions_path):
        with open(precomputed_reconstructions_path, 'rb') as f:
            precomputed_reconstructions = pickle.load(f)
        print('precomputed reconstructions loaded from disk')
        loaded_any = True
    else:
        print(f'missing cache file: {precomputed_reconstructions_path}')

    if os.path.exists(precomputed_latents_path):
        with open(precomputed_latents_path, 'rb') as f:
            precomputed_latents = pickle.load(f)
        print('precomputed latents loaded from disk, shape: ', precomputed_latents.shape)
        l_max = float(np.max(precomputed_latents))
        l_min = float(np.min(precomputed_latents))
        print(f'l_max: {l_max}, l_min: {l_min}')
        loaded_any = True
        if 'latent_im' in globals():
            latent_im.set_clim(vmin=l_min, vmax=l_max)
    else:
        print(f'missing cache file: {precomputed_latents_path}')

    if os.path.exists(precomputed_sequences_path):
        with open(precomputed_sequences_path, 'rb') as f:
            midi_processor.sequences = pickle.load(f)
        print('sequences loaded from disk')
        print('n_sequences: ', len(midi_processor.sequences))
        loaded_any = True
    else:
        print(f'missing cache file: {precomputed_sequences_path}')

    if precomputed_reconstructions is None and getattr(midi_processor, 'sequences', None):
        midi_processor.generate_matrices()
        precomputed_reconstructions = midi_processor.matrices
        print('generated reconstructions from loaded sequences')

    if os.path.exists(control_points_path):
        loaded_control_points = np.load(control_points_path)
        if loaded_control_points.shape == control_points.shape:
            control_points[:, :] = loaded_control_points
            scatter_points.set_offsets(control_points)
            print('control point locations loaded from disk')
            loaded_any = True
        else:
            print(
                f'control point shape mismatch: cached {loaded_control_points.shape}, current {control_points.shape}'
            )
    else:
        print(f'missing cache file: {control_points_path}')

    inferred_resolution = None
    if precomputed_latents is not None:
        inferred_resolution = infer_grid_resolution(len(precomputed_latents))
    elif precomputed_reconstructions is not None:
        inferred_resolution = infer_grid_resolution(len(precomputed_reconstructions))

    if inferred_resolution is not None:
        grid_resolution = inferred_resolution
        last_recalculated_grid_resolution = inferred_resolution
        if 'grid_slider' in globals():
            grid_slider.set_val(inferred_resolution)
        else:
            apply_world_grid(grid_resolution)

    weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)
    nearest_x, nearest_y = None, None

    if loaded_any:
        update_latent_display()
        fig.canvas.draw_idle()
    else:
        print('no cached state loaded')




########################################
# Model and Data Initialization
########################################
data, metadata = load_dataset(hparams)

NUM_POINTS = 5
data, metadata, sample_indices = pick_reproducible_samples(data, metadata, NUM_POINTS, hparams.seed)

print("Data shape:", data.shape)
print("Metadata shape: ", metadata.shape)
print("Sample indices:", sample_indices)


selected_countries = [os.path.basename(meta[0]) for meta in metadata]

point_positions = []
print('num points: ', NUM_POINTS)

latent_vectors = create_latent_vectors(data, model)
print('latent vectors: ', latent_vectors.shape)


# control points in circle:
print('Using circle method initially; cached positions can be loaded via button')
control_points = np.array([
    [np.cos(2 * np.pi * i / NUM_POINTS) * radius_x * 0.8, np.sin(2 * np.pi * i / NUM_POINTS) * radius_y * 0.8]
    for i in range(NUM_POINTS)
])

cursor_pos = np.array([0.0, 0.0])

# Compute initial RBF weights
# weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)


# model.to(torch.device('cuda' if torch.cuda.is_available() else 'gpu'))
########################################
# MIDI Setup
########################################
midi_processor.playback()




########################################
# Plot Setup
########################################
# fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 12))
# plt.use('Qt5Agg')  # Or try 'TkAgg', 'WebAgg', etc.
# Create 3 subplots: Map, Matrix, and Latent Strip
fig = plt.figure(figsize=(10, 10))

# Define a 10-row grid. All plots will span the full width (1 column).
# This allows us to make the signature extremely thin compared to the map.
gs = fig.add_gridspec(20, 1, hspace=0.8)

# Map (Top, takes 6 rows: 0, 1, 2, 3, 4, 5)
ax = fig.add_subplot(gs[0:12, 0])

# Matrix (Middle, takes 3 rows: 6, 7, 8)
ax2 = fig.add_subplot(gs[12:18, 0])

# Latent Strip (Bottom, takes ONLY 1 row: index 9)
# Since it is only 1/10th of the grid height, it will be very narrow.
ax3 = fig.add_subplot(gs[19, 0])

plt.subplots_adjust(left=0.1, right=0.9, top=0.99, bottom=0.1)



ax.set_title("RBF Latent Interpolation")



ax.set_xlim(-radius_x, radius_x)
ax.set_ylim(-y_limit, y_limit)

# --- ADD PNG BACKGROUND ---
try:
    # Load the world.png directly from your project directory
    bg_img = mpimg.imread("assets/world.png")

    # Display it behind everything (zorder=-1)
    # Match 'extent' to your plot limits (-1.5 to 1.5)
    ax.imshow(bg_img, extent=[-radius_x, radius_x, -y_limit, y_limit], zorder=-1, alpha=0.8)
except Exception as e:
    print(f"Could not load world.png background: {e}")
# -------------------------------



# World SVG Grid
tick_font_size = 5


def apply_world_grid(resolution):
    x_ticks = np.linspace(-radius_x, radius_x, resolution + 1)
    y_ticks = np.linspace(-y_limit, y_limit, resolution + 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax.tick_params(axis='both', which='minor', length=0)
    ax.grid(False, which='minor')
    ax.grid(True, which='major', axis='both', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)


apply_world_grid(grid_resolution)

# ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
# ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2)

# MIDI Grid
midi_grid_width = hparams.width * hparams.n_timing
im = ax2.imshow(np.zeros((hparams.height, midi_grid_width)), aspect='auto', cmap='Grays', origin='lower', interpolation='nearest', vmin=0, vmax=1)
ax2.set_title("Reconstructed Sample")
ax2.set_box_aspect(0.4)

ax2.set_xticks(np.arange(-0.5, midi_grid_width, hparams.n_timing), minor=True)
ax2.set_xticks(np.arange(0, midi_grid_width, 4 * hparams.n_timing), minor=False)

# ax.set_xticklabels(np.arange(0, midi_grid_width, 4 * hparams.n_timing) // hparams.n_timing + 1)
ax2.set_yticks(np.arange(-0.5, hparams.height, 1), minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)
ax2.tick_params(axis='both', which='minor', labelsize=tick_font_size)


lane_positions = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
lane_names = [
    "Kick",
    "Rim shot",
    "Snare",
    "Snare",
    "HH Closed",
    "Low Tom",
    "HH Foot",
    "Mid Tom",
    "HH Open",
    "Tom 2 Rim",
    "High Tom",
    "Crash",
    "High Tom Rim",
    "Ride",
]

ax2.set_yticks(lane_positions)
ax2.set_yticklabels(lane_names)

# Setup ax3 for the Latent Vector Strip
ax3.set_title("Latent Vector")
# Create an initial empty strip (1 row, LATENT_DIM columns)
latent_im = ax3.imshow(np.zeros((1, LATENT_DIM)), aspect='auto',
                       cmap='coolwarm', interpolation='bilinear', vmin=l_min, vmax=l_max)
ax3.set_yticks([]) # Hide Y-axis for the strip
ax3.tick_params(axis='x', labelsize=tick_font_size)


# initial_data = np.zeros((hparams.height, hparams.width))  # Or some default image
# im = ax2.imshow(initial_data, aspect='auto', cmap='viridis')  # or whatever parameters you need

# Recalculate Button:
ax_button = plt.axes([0.14, 0.01, 0.14, 0.04])  # Position for button
recalculate_button = Button(ax_button, 'Recalculate Positions')
recalculate_button.label.set_fontsize(6)

# Grid Resolution Slider and Reset Button:
ax_grid_slider = plt.axes([0.33, 0.02, 0.28, 0.03])
grid_slider = Slider(ax_grid_slider, 'Grid', 4, 20, valinit=grid_resolution, valstep=1)
grid_slider.label.set_fontsize(6)
grid_slider.valtext.set_fontsize(6)

ax_reset_grid_button = plt.axes([0.64, 0.01, 0.14, 0.04])
reset_grid_button = Button(ax_reset_grid_button, 'Reset Grid')
reset_grid_button.label.set_fontsize(6)

ax_load_cache_button = plt.axes([0.80, 0.01, 0.14, 0.04])
load_cache_button = Button(ax_load_cache_button, 'Load Cached')
load_cache_button.label.set_fontsize(6)


def set_recalculate_button_state(state):
    labels = {
        'idle': 'Recalculate Positions',
        'running': 'Stop Recalculation',
        'stopping': 'Stopping...',
    }
    facecolors = {
        'idle': '0.85',
        'running': '#f2c9c9',
        'stopping': '#f0dfb3',
    }
    recalculate_button.label.set_text(labels[state])
    ax_button.set_facecolor(facecolors[state])
    fig.canvas.draw_idle()


set_recalculate_button_state('idle')

# Reinitialize latent Vectors:
# ax_button2 = plt.axes([0.31, 0.01, 0.14, 0.04])
# reinit_button = Button(ax_button2, 'Change Base Rhythms')
# reinit_button.label.set_fontsize(6)

scatter_points = ax.scatter(control_points[:, 0], control_points[:, 1], color='blue', s=30)
# Add text labels
# point_labels = []
# for i, (x, y) in enumerate(control_points):
#     txt = ax.text(x + 0.03, y + 0.03, str(selected_countries[i]), color='black', fontsize=6, weight='bold')
#     point_labels.append(txt)


# Create a SINGLE shared hover label (initially invisible)
hover_label = ax.text(0, 0, "", color='black', fontsize=7, weight='bold',
                      visible=False, zorder=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))


def on_hover(event):
    if event.inaxes != ax:
        return

    cont, ind = scatter_points.contains(event)

    if cont:
        # Get the index of the point being hovered
        idx = ind['ind'][0]
        # Get the data coordinates of that specific point
        pos = scatter_points.get_offsets()[idx]

        # Update label: position (slightly offset), text content, and visibility
        hover_label.set_position((pos[0] + 0.05, pos[1] + 0.05))
        hover_label.set_text(selected_countries[idx])
        hover_label.set_visible(True)
    else:
        hover_label.set_visible(False)

    fig.canvas.draw_idle()


# Connect the hover event
fig.canvas.mpl_connect("motion_notify_event", on_hover)



scatter_cursor = ax.scatter(cursor_pos[0], cursor_pos[1], color='red', s=100, marker='x')
# latent_text = ax.text(-1.2, 1.4, "", fontsize=7, family="monospace")
display_latent_text = False
if display_latent_text:
    latent_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=7, family="monospace", verticalalignment='top')

influences_text = ax.text(0.05, -0.07, "", transform=ax.transAxes, fontsize=7, family="monospace", verticalalignment='top')
def nearest_cell_center(value, vmin, vmax, n_cells):
    cell_size = (vmax - vmin) / n_cells
    idx = int((value - vmin) / cell_size)
    idx = max(0, min(n_cells - 1, idx))
    center = vmin + (idx + 0.5) * cell_size
    return idx, center
nearest_x, nearest_y = None, None

def get_centers(lin):
    centers = []
    for i, x in enumerate(lin[:-1]):
        # print(i, x)
        dist = lin[i + 1] - lin[i]
        start = lin[i]
        shift = dist / 2
        center = start + shift
        centers.append(center)
    return centers


def show_matrix_debug_preview(note_matrix):
    """Show a simple, robust pixel preview of the current matrix in an OpenCV window."""
    if not CV2_AVAILABLE or note_matrix is None:
        return

    mat = np.asarray(note_matrix, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=0.0)

    max_val = float(np.max(mat)) if mat.size else 0.0
    min_val = float(np.min(mat)) if mat.size else 0.0
    if max_val > 1.0 or min_val < 0.0:
        denom = max(max_val - min_val, 1e-8)
        mat = (mat - min_val) / denom
    else:
        mat = np.clip(mat, 0.0, 1.0)

    # Match matplotlib's origin='lower' by flipping vertically for OpenCV display.
    img = np.flipud((mat * 255.0).astype(np.uint8))

    scale = 16
    preview = cv2.resize(
        img,
        (img.shape[1] * scale, img.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    cv2.imshow('MIDI Matrix Preview', preview)
    cv2.waitKey(1)

def update_latent_display(isCursor=False):
    global weights, nearest_x, nearest_y, grid_resolution, im, note_matrix, precomputed_reconstructions, precomputed_latents

    # 1. Calculate the raw basis values (Phi) for the current cursor position
    # This uses the exact same math as your rbf_interpolate_exact function
    dists = np.linalg.norm(control_points - cursor_pos, axis=1)
    phi_values = np.exp(-(dists ** 2) / (2 * RBF_SIGMA ** 2))

    # 2. Normalize these values to get a percentage (0.0 to 1.0)
    # This represents how much "of Point X" is in the current sound
    total_phi = np.sum(phi_values)
    influence_weights = phi_values / total_phi if total_phi > 0 else phi_values

    # 3. Create a HUD display string for the top influencers
    top_indices = np.argsort(influence_weights)[::-1][:5]  # Top 3
    influence_str = " | ".join([
        f"{selected_countries[i][:12]}: {influence_weights[i] * 100:.1f}%"
        for i in top_indices if influence_weights[i] > 0.01
    ])
    influences_text.set_text(f"Top Influencers: [{influence_str}]")


    if display_latent_text:
        latent = rbf_interpolate_exact(control_points, weights, cursor_pos, sigma=RBF_SIGMA)
        latent_str = np.array2string(np.round(latent, 2), precision=3, separator=",")
        latent_text.set_text(f"Interpolated latent: {latent_str[:30]}...]")


    if isCursor:
        expected_count = grid_resolution * grid_resolution
        recon_count = 0 if precomputed_reconstructions is None else len(precomputed_reconstructions)
        latent_count = 0 if precomputed_latents is None else len(precomputed_latents)
        if recon_count != expected_count or latent_count != expected_count:
            return

        # Retrieve reconstruction from the precomputed buffer
        # grid_resolution = 8
        
        x_vals = np.linspace(-radius_x, radius_x, grid_resolution + 1)
        y_vals = np.linspace(-y_limit, y_limit, grid_resolution + 1)
        x_vals = np.array(get_centers(x_vals))
        y_vals = np.array(get_centers(y_vals))

        # print(cursor_pos, nearest_x, nearest_y)

        # Find the nearest grid point to the cursor position
        current_nearest_x = min(x_vals, key=lambda x: abs(x - cursor_pos[0]))
        current_nearest_y = min(y_vals, key=lambda y: abs(y - cursor_pos[1]))
        # x_idx, current_nearest_x = nearest_cell_center(cursor_pos[0], -radius_x, radius_x, grid_resolution)
        # y_idx, current_nearest_y = nearest_cell_center(cursor_pos[1], -y_limit, y_limit, grid_resolution)
        if current_nearest_x != nearest_x or current_nearest_y != nearest_y:
            # find the original idx in precomputed token sequences stored in midi_processor.sequences
            x_idx = np.argmin(np.abs(x_vals - current_nearest_x))
            y_idx = np.argmin(np.abs(y_vals - current_nearest_y))
            original_idx = x_idx * grid_resolution + y_idx


            # original_idx = y_idx * grid_resolution + x_idx
            midi_processor.seq_index = original_idx

            # print(current_nearest_x, current_nearest_y)
            nearest_x = current_nearest_x
            nearest_y = current_nearest_y

            sample_mat = precomputed_reconstructions[original_idx]
            latent = precomputed_latents[original_idx]



            # Update the ax3 visual strip
            latent_im.set_data(latent.reshape(1, -1))
            # Optional: auto-scale colors based on the current vector values
            # latent_im.set_clim(vmin=latent.min(), vmax=latent.max())

            # note_matrix = sample_mat.copy()

            # mask = note_matrix != 0
            # note_matrix[mask] = note_matrix[mask] * 0.7 + 0.3
            
            # ax2.imshow(sample_mat, aspect='auto', cmap='Grays', origin='lower')
            im.set_data(sample_mat)
            # show_matrix_debug_preview(sample_mat)
            # im.set_clim(vmin=sample_mat.min(), vmax=sample_mat.max())  # <-- Add this line
            # make sample_mat visible with opencv2:
            # cv2.imshow('Sample Matrix', sample_mat)
            # cv2.waitKey(1)


    fig.canvas.draw_idle()
    # plt.draw()

# Draggable blue dots
draggables = []
for i in range(NUM_POINTS):
    d = DraggablePoint(control_points[i], i, scatter_points, update_latent_display, fig, ax, control_points)
    d.connect()
    draggables.append(d)

# Draggable red cursor
cursor_control = CursorController(cursor_pos, scatter_cursor, update_latent_display, fig, ax, control_points)
cursor_control.connect()

def recalculate_worker(control_points_snapshot, latent_vectors_snapshot, resolution, stop_event):
    global recalc_status, recalc_result, recalc_error

    start = time.time()
    print('Recomputing RBF weights...')

    try:
        worker_weights = compute_rbf_weights(control_points_snapshot, latent_vectors_snapshot, sigma=RBF_SIGMA)

        x_vals = np.linspace(-radius_x, radius_x, resolution + 1)
        y_vals = np.linspace(-y_limit, y_limit, resolution + 1)
        x_vals = get_centers(x_vals)
        y_vals = get_centers(y_vals)

        print()
        print('Precomputing reconstructions for grid points...')
        latents = np.zeros((resolution * resolution, LATENT_DIM))
        counter = 0
        for x in x_vals:
            for y in y_vals:
                if stop_event.is_set():
                    with recalc_lock:
                        recalc_status = 'cancelled'
                    return
                cursor_pos_grid = np.array([x, y])
                latents[counter] = rbf_interpolate_exact(control_points_snapshot, worker_weights, cursor_pos_grid, sigma=RBF_SIGMA)
                counter += 1

        print(f'Processing {latents.shape[0]} latent vectors...')
        z = torch.tensor(latents, dtype=torch.float32)
        sequences = []
        with torch.no_grad():
            for start_idx in range(0, latents.shape[0], recalc_batch_size):
                if stop_event.is_set():
                    with recalc_lock:
                        recalc_status = 'cancelled'
                    return
                end_idx = min(start_idx + recalc_batch_size, latents.shape[0])
                samples = model.sample(z[start_idx:end_idx], hparams.max_seq_len, 0.1, False)
                sequences.extend(samples.tolist())

        if stop_event.is_set():
            with recalc_lock:
                recalc_status = 'cancelled'
            return

        worker_midi_processor = MidiProcessor(hparams)
        worker_midi_processor.sequences = sequences
        worker_midi_processor.generate_matrices()

        with recalc_lock:
            recalc_result = {
                'weights': worker_weights,
                'latents': latents,
                'sequences': sequences,
                'reconstructions': worker_midi_processor.matrices,
                'resolution': resolution,
                'duration': time.time() - start,
            }
            recalc_error = None
            recalc_status = 'done'
    except Exception as exc:
        with recalc_lock:
            recalc_error = exc
            recalc_result = None
            recalc_status = 'error'


def apply_recalculation_result(result):
    global weights, precomputed_reconstructions, precomputed_latents, nearest_x, nearest_y
    global grid_resolution, last_recalculated_grid_resolution

    weights = result['weights']
    midi_processor.sequences = result['sequences']
    midi_processor.matrices = result['reconstructions']
    precomputed_reconstructions = result['reconstructions']
    precomputed_latents = result['latents']
    nearest_x, nearest_y = None, None
    grid_resolution = result['resolution']
    last_recalculated_grid_resolution = result['resolution']

    if int(round(grid_slider.val)) != result['resolution']:
        grid_slider.set_val(result['resolution'])
    else:
        apply_world_grid(grid_resolution)

    print(f"Precomputation complete! {result['duration']:.2f}s")
    update_latent_display()


def poll_recalculation_state():
    global recalc_thread, recalc_stop_event, recalc_status, recalc_result, recalc_error

    with recalc_lock:
        status = recalc_status
        result = recalc_result
        error = recalc_error

    if status == 'done':
        apply_recalculation_result(result)
        with recalc_lock:
            recalc_status = 'idle'
            recalc_result = None
            recalc_error = None
        recalc_thread = None
        recalc_stop_event = None
        set_recalculate_button_state('idle')
    elif status == 'cancelled':
        print('Recalculation cancelled.')
        with recalc_lock:
            recalc_status = 'idle'
            recalc_result = None
            recalc_error = None
        recalc_thread = None
        recalc_stop_event = None
        set_recalculate_button_state('idle')
    elif status == 'error':
        print(f'Recalculation failed: {error}')
        with recalc_lock:
            recalc_status = 'idle'
            recalc_result = None
            recalc_error = None
        recalc_thread = None
        recalc_stop_event = None
        set_recalculate_button_state('idle')
    return True


def start_recalculation_thread():
    global recalc_thread, recalc_stop_event, recalc_status, recalc_result, recalc_error

    control_points_snapshot = control_points.copy()
    latent_vectors_snapshot = latent_vectors.copy()
    resolution_snapshot = grid_resolution

    with recalc_lock:
        recalc_status = 'running'
        recalc_result = None
        recalc_error = None

    recalc_stop_event = threading.Event()
    recalc_thread = threading.Thread(
        target=recalculate_worker,
        args=(control_points_snapshot, latent_vectors_snapshot, resolution_snapshot, recalc_stop_event),
        daemon=True,
    )
    recalc_thread.start()
    set_recalculate_button_state('running')


def recalculate_and_update(event):
    global recalc_status

    thread_running = recalc_thread is not None and recalc_thread.is_alive()
    if thread_running:
        if recalc_stop_event is not None and not recalc_stop_event.is_set():
            recalc_stop_event.set()
            with recalc_lock:
                recalc_status = 'stopping'
            set_recalculate_button_state('stopping')
        return

    start_recalculation_thread()


# Connect button to recalculate_and_update function
def on_button_click(event):
    recalculate_and_update(None)


def on_grid_slider_change(val):
    global grid_resolution, nearest_x, nearest_y
    grid_resolution = int(round(val))
    nearest_x, nearest_y = None, None
    apply_world_grid(grid_resolution)
    fig.canvas.draw_idle()


def on_reset_grid_click(event):
    grid_slider.set_val(last_recalculated_grid_resolution)


def on_load_cache_click(event):
    load_cached_state()



recalculate_button.on_clicked(on_button_click)
grid_slider.on_changed(on_grid_slider_change)
reset_grid_button.on_clicked(on_reset_grid_click)
load_cache_button.on_clicked(on_load_cache_click)

recalc_timer = fig.canvas.new_timer(interval=150)
recalc_timer.add_callback(poll_recalculation_state)
recalc_timer.start()

# Initial display
update_latent_display()
plt.show()
midi_processor.stop_playback()
if CV2_AVAILABLE:
    cv2.destroyAllWindows()

