import numpy as np
import matplotlib.pyplot as plt
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize
from modules.midi_processor import MidiProcessor
import torch
from modules.utilities import seq2mat
from matplotlib.widgets import Button
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

# Find the latest model in the models directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
model_files = glob.glob(os.path.join(_script_dir, '../models/model_*.pt'))
if not model_files:
    raise FileNotFoundError("No model files found in ../models/ directory")
latest_model = max(model_files, key=os.path.getmtime)
print(f'Loading model: {latest_model}')

checkpoint = torch.load(latest_model, map_location='cpu')
model_state_dict = checkpoint['model_state_dict']
hparams_dict = checkpoint['hparams']

# Recreate the hparams object
hparams = SimpleNamespace(**hparams_dict)


model = TransformerVAE(hparams)
model.load_state_dict(model_state_dict)
model.eval()
print('--' * 10)
print('model config: ')
print('model dimensions: ', hparams.d_model)
print('n_layers: ', hparams.n_layers)
print('n_heads: ', hparams.n_heads)
print('latent dim: ', hparams.latent_dim)
print('--' * 10)

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

# Example: Print a few country positions to verify
# print("\nParsed Country Coordinates (mapped):")
# for country in ["Afghanistan", "Angola", "Germany"]:
#     if country in country_coords:
#         print(f"{country}: {np.round(country_coords[country], 3)}")
# exit()
# for name, coords in list(country_coords.items())[:10]:
#     print(f"{name}: {coords}")
country_list = list(country_coords.keys())
selected_countries = ['United States', 'Spain',  'New Zealand', 'China', 'Australia', 'Togo', 'Namibia', 'Chile', 'Germany']

# NUM_POINTS = len(selected_countries)  # number of control points / latent vectors
LATENT_DIM = hparams.latent_dim  # dimension of latent space
RBF_SIGMA = 0.3  # width of Gaussian RBF
grid_resolution = 16

# Global buffer to store precomputed values
precomputed_reconstructions = None
precomputed_latents = None
l_min, l_max = None, None
if os.path.exists('outputs/precomputed_reconstructions.pkl'):
    with open('outputs/precomputed_reconstructions.pkl', 'rb') as f:
        precomputed_reconstructions = pickle.load(f)
        print('precomputed reconstructions loaded from disk')
if os.path.exists('outputs/precomputed_latents.pkl'):
    with open('outputs/precomputed_latents.pkl', 'rb') as f:
        precomputed_latents = pickle.load(f)
        print('precomputed latents loaded from disk, shape: ', precomputed_latents.shape)
        
        l_max = float(np.max(precomputed_latents))
        l_min = float(np.min(precomputed_latents))
        print(f'l_max: {l_max}, l_min: {l_min}')

midi_processor = MidiProcessor(hparams)
if os.path.exists('outputs/precomputed_sequences.pkl'):
    with open('outputs/precomputed_sequences.pkl', 'rb') as f:
        midi_processor.sequences = pickle.load(f)
        print('sequences loaded from disk')
        print('n_sequences: ', len(midi_processor.sequences))
########################################
# Exact RBF Interpolation
########################################


def compute_rbf_weights(control_points, latent_vectors, sigma=0.7, lambd=1e-4):
    """
    Computes RBF weights with Ridge Regularization (lambda) to handle
    many/crowded control points without numerical instability.
    """
    control_points = np.asarray(control_points)
    N = len(control_points)

    # Distance matrix
    dists = np.linalg.norm(control_points[:, None, :] - control_points[None, :, :], axis=2)

    # Gaussian RBF
    Phi = np.exp(-(dists ** 2) / (2 * sigma ** 2))

    # Regularization: Add a tiny bit to the diagonal (Ridge Regression)
    # This keeps the matrix invertible even if points are very close.
    Phi += lambd * np.eye(N)

    # Solve Phi @ W = latent_vectors
    weights = np.linalg.solve(Phi, latent_vectors)
    return weights


# def compute_rbf_weights(control_points, latent_vectors, sigma=1.0):
#     """
#     Computes RBF weights such that interpolation exactly reproduces latent vectors at control points.
#     """
#     N = len(control_points)
#     # Distance matrix
#     dists = np.linalg.norm(control_points[:, None, :] - control_points[None, :, :], axis=2)
#     # Gaussian RBF
#     Phi = np.exp(-(dists ** 2) / (2 * sigma ** 2))
#     # Solve Phi @ W = latent_vectors
#     weights = np.linalg.solve(Phi, latent_vectors)
#     return weights


def rbf_interpolate_exact(control_points, weights, cursor_pos, sigma=1.0):
    """
    Interpolates latent vector at cursor_pos using precomputed RBF weights.
    """
    dists = np.linalg.norm(control_points - cursor_pos, axis=1)
    rbf_vals = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    return rbf_vals @ weights


########################################
# Draggable Point
########################################

class DraggablePoint:
    def __init__(self, point, index, scatter, update_callback):
        self.point = point
        self.index = index
        self.scatter = scatter
        self.update_callback = update_callback
        self.press = None

    def connect(self):
        self.cidpress = fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = fig.canvas.mpl_connect("motion_notify_event", self.on_motion)


    def on_press(self, event):
        if event.inaxes != ax:
            return
        contains, _ = self.scatter.contains(event)
        if contains:
            # Find the nearest point to click
            dists = np.linalg.norm(control_points - np.array([event.xdata, event.ydata]), axis=1)
            idx = np.argmin(dists)
            if np.allclose(control_points[idx], self.point):
                self.press = (self.point[0], self.point[1], event.xdata, event.ydata)

    def on_release(self, event):
        self.press = None


    def on_motion(self, event):
        if self.press is None or event.inaxes != ax:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point[0] = x0 + dx
        self.point[1] = y0 + dy
        self.scatter.set_offsets(control_points)
        # Update the corresponding label position
        # point_labels[self.index].set_position((self.point[0] + 0.03, self.point[1] + 0.03))
        self.update_callback()


########################################
# Cursor Controller
########################################

class CursorController:
    def __init__(self, cursor, scatter, update_callback):
        self.cursor = cursor
        self.scatter = scatter
        self.update_callback = update_callback
        self.press = None

    def connect(self):
        self.cidpress = fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cidrelease = fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cidmotion = fig.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def on_press(self, event):
        if event.inaxes != ax:
            return
        contains, _ = self.scatter.contains(event)
        if contains:
            self.press = (self.cursor[0], self.cursor[1], event.xdata, event.ydata)

    def on_release(self, event):
        self.press = None

    def on_motion(self, event):
        if self.press is None or event.inaxes != ax:
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.cursor[0] = x0 + dx
        self.cursor[1] = y0 + dy
        self.scatter.set_offsets([self.cursor])
        self.update_callback(isCursor=True)


########################################
# Model and Data Initialization
########################################
processor = MidiProcessor(hparams)
if os.path.exists('data/midis.npz'):
    print("Loading preprocessed dataset from midis.npz")
    npz_data = np.load('data/midis.npz', allow_pickle=True)
    data = npz_data['data']
    metadata = npz_data['metadata']
    # processor.sequences = data
else:

    abs_path = 'midifiles'

    processor.extract(abs_path, 0, 0)
    # processor.visualize(playback=False)
    data = processor.convert_to_numpy()
    metadata = processor.metadata

    filtered_data = []
    filtered_metadata = []
    for i in range(len(metadata)):
        if metadata[i][1] == 1:
            filtered_data.append(data[i])
            filtered_metadata.append(metadata[i])
    # print("Filtered data shape: (", len(filtered_data), ', 256)')
    # print("Filtered metadata shape: (", len(filtered_metadata), ', 2)')

    data = np.array(filtered_data)
    metadata = np.array(filtered_metadata)
    # processor.sequences = data


    np.savez('data/midis.npz', data=data, metadata=metadata)
    print("Saved preprocessed dataset to data/midis.npz")
print("Data shape:", data.shape)
print("Metadata shape: ", metadata.shape)


print(f"Original n_samples: {len(data)}")
# print(f"Unique filenames: {len(seen_filenames)}")
# print(f"Filtered data shape: {filtered_data.shape}")

selected_countries = [os.path.basename(meta[0]) for meta in metadata]
NUM_POINTS = len(selected_countries)
point_positions = []
print('num points: ', NUM_POINTS)



def create_latent_vectors(input_data: np.ndarray):
    data_tensor = torch.from_numpy(input_data).long()
    logits, mu, log_var = model(data_tensor)
    z = mu.detach().numpy()
    return z

latent_vectors = create_latent_vectors(data)
print('latent vectors: ', latent_vectors.shape)


# control points in circle:
if os.path.exists('outputs/control_points.npy'):
    print("Loading control point locations from control_points.npy")
    control_points = np.load('outputs/control_points.npy')
    # control_points *= scale
else:
    print("control_points.npy not found, using circle method")
    control_points = np.array([
        [np.cos(2 * np.pi * i / NUM_POINTS) * radius_x * 0.8, np.sin(2 * np.pi * i / NUM_POINTS) * radius_y * 0.8]
        for i in range(NUM_POINTS)
    ])
# print('control points: ', control_points)
# control_points = np.random.uniform(-1, 1, size=(NUM_POINTS, 2))
# control points at selected countries:
# Assuming selected_countries is a list like ["Germany", "Brazil", "Japan"]
# control_points = np.array([country_coords[country] for country in selected_countries if country in country_coords])

# latent_vectors = np.random.normal(size=(NUM_POINTS, LATENT_DIM))
cursor_pos = np.array([0.0, 0.0])
# print('\nLatent Vectors:')
# for i, vec in enumerate(latent_vectors):
#     vec_string = np.round(vec[:5], 2)
#     vec_string = str(vec_string)[0:-1] + ' ...]'
#     print(i, vec_string)

# Compute initial RBF weights
weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)


# model.to(torch.device('cuda' if torch.cuda.is_available() else 'gpu'))
########################################
# MIDI Setup
########################################
# playback_position = 0
# bpm = 100
# beat_length = 60 / bpm
# sixteenth_note_length = beat_length / 4
# midi_output_port_name = mido.get_output_names()[0]
# print('using midi port ', midi_output_port_name)
# output_port = mido.open_output(midi_output_port_name, callback=lambda msg: print(msg))
# note_matrix = np.zeros((hparams.height, hparams.width))
# stop_event = threading.Event()
#
# def play_midi():
#     global playback_position, note_matrix, current_index
#
#     vel = 0
#     while not stop_event.is_set():
#         matrix = note_matrix
#         matrix = np.array(matrix)
#         # print(playback_position)
#
#         for pitch in range(hparams.height):
#             if matrix[pitch, playback_position] > 0:
#                 vel = int(matrix[pitch, playback_position] * 127)
#                 # print(pitch + 36)
#                 output_port.send(mido.Message('note_on', note=pitch + 36, velocity=vel))
#         time.sleep(sixteenth_note_length)
#         for i in range(hparams.height):
#             output_port.send(mido.Message('note_off', note=i + 36, velocity=vel))
#         playback_position += 1
#         playback_position %= hparams.width
#
# stop_event.clear()  # Ensure the event is not set before starting
# thread = threading.Thread(target=play_midi)
# thread.start()

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
grid_arrange_x = np.linspace(-radius_x, radius_x, grid_resolution + 1)  # Ensures even spacing including endpoints
grid_arrange_y = np.linspace(-y_limit, y_limit, grid_resolution + 1)
print(grid_arrange_x)
tick_font_size = 5
ax.set_xticks(grid_arrange_x)
ax.set_yticks(grid_arrange_y, minor=True)
# Turn on the grid for BOTH major and minor ticks
# 'which='minor'' is the key here to see your linspace grid
ax.grid(visible=True, which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
# ax.grid(axis='x', visible=True, which='major', color='black', linestyle='-', linewidth=0.3, alpha=0.3)
ax.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)

ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_font_size)

ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)



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
nearest_x, nearest_y = 0,0

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
    d = DraggablePoint(control_points[i], i, scatter_points, update_latent_display)
    d.connect()
    draggables.append(d)

# Draggable red cursor
cursor_control = CursorController(cursor_pos, scatter_cursor, update_latent_display)
cursor_control.connect()

def recalculate_and_update(event):
    global weights, latent_vectors, control_points, cursor_pos, precomputed_reconstructions, grid_resolution, precomputed_latents
    # print('control points: ', control_points)
    # Recompute weights
    start = time.time()
    weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)

    # Define the resolution of the 2D space grid
    # grid_resolution = 8  # Higher value means finer grid
    x_vals = np.linspace(-radius_x, radius_x, grid_resolution + 1)
    y_vals = np.linspace(-y_limit, y_limit, grid_resolution + 1)
    x_vals = get_centers(x_vals)
    y_vals = get_centers(y_vals)


    # Initialize a buffer for precomputed reconstructions
    precomputed_reconstructions = []
    precomputed_latents = []
    print()
    counter = 0
    # Iterate over all grid points
    latents = np.zeros((grid_resolution*grid_resolution, LATENT_DIM))
    for x in x_vals:
        for y in y_vals:
            cursor_pos_grid = np.array([x, y])
            # print(counter, cursor_pos_grid)
            latent = rbf_interpolate_exact(control_points, weights, cursor_pos_grid, sigma=RBF_SIGMA)
            latents[counter] = latent
            counter += 1


    # # Reconstruct sequences using the decoder
    print(f'Processing {latents.shape[0]} latent vectors...')
    z = torch.tensor(latents, dtype=torch.float32)
    # print(z.shape)
    with torch.no_grad():
        samples = model.sample(z, hparams.max_seq_len, 0.1, False)
        print(f'Processing {samples.shape[0]} sequences...')
    sequences = samples.tolist()

    midi_processor.sequences = sequences
    midi_processor.generate_matrices()
    precomputed_reconstructions = midi_processor.matrices
    
    process_time = time.time()-start

    # save point locations and rhythms:
    # reconstructions from the model:
    print(f"Precomputation complete! {process_time:.2f}s")
    with open('outputs/precomputed_reconstructions.pkl', 'wb') as f:
        pickle.dump(precomputed_reconstructions, f)
    print('saved reconstructions as ', 'precomputed_reconstructions.pkl')
    # locations of control_points
    np.save('outputs/control_points.npy', control_points)
    print('saved control point locations')
    # save the latent vectors in a separate numpy file:
    with open('outputs/precomputed_latents.pkl', 'wb') as f:
        pickle.dump(latents, f)
    print('saved latent vectors')
    # save midi_processor.sequences
    with open('outputs/precomputed_sequences.pkl', 'wb') as f:
        pickle.dump(midi_processor.sequences, f)
    print('saved midi sequences')


# recalculate_and_update(None)





# Connect button to recalculate_and_update function
def on_button_click(event):
    recalculate_and_update(None)

# def reinitialize(event):
#     global latent_vectors, max_length, note_matrix, nearest_x, nearest_y
#     latent_vectors = create_latent_vectors(data)
#     latent_vectors = np.array(latent_vectors)
#     print('\nLatent Vectors:')
#     for i, vec in enumerate(latent_vectors):
#         print(i, np.round(vec[:5], 2))
#     recalculate_and_update(None)
#     note_matrix = precomputed_reconstructions[(nearest_x, nearest_y)]
#
#     im.set_data(note_matrix)
#     im.set_clim(vmin=note_matrix.min(), vmax=note_matrix.max())
#     fig.canvas.draw_idle()

recalculate_button.on_clicked(on_button_click)
# reinit_button.on_clicked(reinitialize)

# Initial display
update_latent_display()
plt.show()
midi_processor.stop_playback()
if CV2_AVAILABLE:
    cv2.destroyAllWindows()
# stop_event.set()
# thread.join()
# print('control: ', control_points)
