import numpy as np
import matplotlib.pyplot as plt
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize
import torch
from modules.utilities import seq2mat
from matplotlib.widgets import Button
import time
import mido
import threading
from types import SimpleNamespace

"""
calculating RBF (Radial Basis Function) interpolation between multiple latent vectors

"""
##########################
# Load model with metadata
##########################

# Load the saved data
checkpoint = torch.load('model_256.pt')
model_state_dict = checkpoint['model_state_dict']
hparams_dict = checkpoint['hparams']

# Recreate the hparams object
hparams = SimpleNamespace(**hparams_dict)


model = TransformerVAE(hparams)
model.load_state_dict(model_state_dict)
model.eval()
########################################
# CONFIGURATION
########################################

NUM_POINTS = 4  # number of control points / latent vectors
LATENT_DIM = hparams.latent_dim  # dimension of latent space
RBF_SIGMA = 0.7  # width of Gaussian RBF
grid_resolution = 16

########################################
# Exact RBF Interpolation
########################################

def compute_rbf_weights(control_points, latent_vectors, sigma=1.0):
    """
    Computes RBF weights such that interpolation exactly reproduces latent vectors at control points.
    """
    N = len(control_points)
    # Distance matrix
    dists = np.linalg.norm(control_points[:, None, :] - control_points[None, :, :], axis=2)
    # Gaussian RBF
    Phi = np.exp(-(dists ** 2) / (2 * sigma ** 2))
    # Solve Phi @ W = latent_vectors
    weights = np.linalg.solve(Phi, latent_vectors)
    return weights


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
            # Find nearest point to click
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
        point_labels[self.index].set_position((self.point[0] + 0.03, self.point[1] + 0.03))
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


# data = np.load('dataset.npz')['tensor']
data = np.load('midi_clips_data.npz')['tensor']



def create_latent_vectors(data):
    selection = [40, 70, 196, 140, 8, 266, 108, 66, 102, 366, 598, 715, 204, 110]
    # l = [266, 108, 66, 102, 366, 598, 715, 204, 110]
    # data_tensor = data[np.random.choice(data.shape[0], NUM_POINTS, replace=False)]
    init_idx = np.random.choice(selection, NUM_POINTS, replace=False)
    # data_tensor = data[init_idx]
    data_tensor = data[:5]

    data_tensor = torch.tensor(data_tensor, dtype=torch.long)
    latent_space_representations = []
    max_length = 0
    for seq in data_tensor:
        # print(seq.shape)
        # exit()
        with torch.no_grad():
            model_input = seq[seq != 0].view(1, -1)  # remove zeros and reshape
            model_input[model_input >= 60] = 59
            # print(model_input.shape)
            # exit()
            logits, mu, logvar = model(model_input)
            z = reparameterize(mu, logvar)

        latent_space_representations.append(z.view(-1))
        max_length = max(max_length, z.size(1))  # track maximum seq length
    return np.array(latent_space_representations), max_length


latent_vectors, max_length = create_latent_vectors(data)
latent_vectors = np.array(latent_vectors)
radius = 1.3
control_points = np.array([
    [np.cos(2 * np.pi * i / NUM_POINTS)*radius, np.sin(2 * np.pi * i / NUM_POINTS)*radius]
    for i in range(NUM_POINTS)
])
# control_points = np.random.uniform(-1, 1, size=(NUM_POINTS, 2))

# latent_vectors = np.random.normal(size=(NUM_POINTS, LATENT_DIM))
cursor_pos = np.array([0.0, 0.0])
print('\nLatent Vectors:')
for i, vec in enumerate(latent_vectors):
    vec_string = np.round(vec[:5], 2)
    vec_string = str(vec_string)[0:-1] + ' ...]'
    print(i, vec_string)

# Compute initial RBF weights
weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)
# Global buffer to store precomputed values
precomputed_reconstructions = None

# model.to(torch.device('cuda' if torch.cuda.is_available() else 'gpu'))
########################################
# MIDI Setup
########################################
playback_position = 0
bpm = 100
beat_length = 60 / bpm
sixteenth_note_length = beat_length / 4
midi_output_port_name = mido.get_output_names()[0]
print('using midi port ', midi_output_port_name)
output_port = mido.open_output(midi_output_port_name, callback=lambda msg: print(msg))
note_matrix = np.zeros((hparams.height, hparams.width))
stop_event = threading.Event()

def play_midi():
    global playback_position, note_matrix, current_index

    vel = 0
    while not stop_event.is_set():
        matrix = note_matrix
        matrix = np.array(matrix)
        # print(playback_position)

        for pitch in range(hparams.height):
            if matrix[pitch, playback_position] > 0:
                vel = int(matrix[pitch, playback_position] * 127)
                # print(pitch + 36)
                output_port.send(mido.Message('note_on', note=pitch + 36, velocity=vel))
        time.sleep(sixteenth_note_length)
        for i in range(hparams.height):
            output_port.send(mido.Message('note_off', note=i + 36, velocity=vel))
        playback_position += 1
        playback_position %= hparams.width

stop_event.clear()  # Ensure the event is not set before starting
thread = threading.Thread(target=play_midi)
thread.start()

########################################
# Plot Setup
########################################
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# plt.use('Qt5Agg')  # Or try 'TkAgg', 'WebAgg', etc.
ax.set_title("Exact RBF Latent Interpolation")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
# grid_arrange = np.arange(-1.5, 1.5, 3/grid_resolution)
grid_arrange = np.linspace(-1.5, 1.5, grid_resolution + 1)  # Ensures even spacing including endpoints
# print(grid_arrange)
tick_font_size = 5
ax.set_xticks(grid_arrange, minor=True)
ax.set_yticks(grid_arrange, minor=True)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2)
ax.axvline(x=0, color='gray', linestyle='-', alpha=0.2)
ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_font_size)

ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)

im = ax2.imshow(np.zeros((hparams.height, hparams.width)), aspect='auto', cmap='Grays', origin='lower')
ax2.set_title("Reconstructed Sample")
ax2.set_xticks(np.arange(-0.5, hparams.width, 1), minor=True)
ax2.set_xticks(np.arange(0, hparams.width, 4))
ax2.set_yticks(np.arange(-0.5, hparams.height, 1), minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
ax2.tick_params(axis='both', which='major', labelsize=tick_font_size)
ax2.tick_params(axis='both', which='minor', labelsize=tick_font_size)

# initial_data = np.zeros((hparams.height, hparams.width))  # Or some default image
# im = ax2.imshow(initial_data, aspect='auto', cmap='viridis')  # or whatever parameters you need

# Recalculate Button:
ax_button = plt.axes([0.14, 0.01, 0.14, 0.04])  # Position for button
recalculate_button = Button(ax_button, 'Recalculate Positions')
recalculate_button.label.set_fontsize(6)

# Reinitialize latent Vectors:
ax_button2 = plt.axes([0.31, 0.01, 0.14, 0.04])
reinit_button = Button(ax_button2, 'Change Base Rhythms')
reinit_button.label.set_fontsize(6)

scatter_points = ax.scatter(control_points[:, 0], control_points[:, 1], color='blue', s=80)
# Add text labels
point_labels = []
for i, (x, y) in enumerate(control_points):
    txt = ax.text(x + 0.03, y + 0.03, str(i), color='black', fontsize=10, weight='bold')
    point_labels.append(txt)

scatter_cursor = ax.scatter(cursor_pos[0], cursor_pos[1], color='red', s=120, marker='x')
latent_text = ax.text(-1.2, 1.4, "", fontsize=7, family="monospace")

nearest_x, nearest_y = 0,0
def update_latent_display(isCursor=False):
    global weights, nearest_x, nearest_y, grid_resolution, im, note_matrix
    # Recompute RBF weights if control points moved
    weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)
    latent = rbf_interpolate_exact(control_points, weights, cursor_pos, sigma=RBF_SIGMA)
    latent_str = np.array2string(np.round(latent, 2), precision=3, separator=",")
    latent_text.set_text(f"Interpolated latent: {latent_str[:30]}...]")

    # # reconstruct
    # z = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
    # sample = model.sample(z, hparams.max_length, 0.1, False)
    # sample_mat = seq2mat(sample.squeeze(0).tolist())
    # # ax2.clear()
    # ax2.imshow(sample_mat, aspect='auto', cmap='Grays', origin='lower')

    if isCursor:
        # Retrieve reconstruction from the precomputed buffer
        # grid_resolution = 8
        
        x_vals = np.linspace(-1.5, 1.5, grid_resolution)
        y_vals = np.linspace(-1.5, 1.5, grid_resolution)
        # offset =  (3/grid_resolution)

        # Find the nearest grid point to the cursor position
        current_nearest_x = min(x_vals, key=lambda x: abs(x - cursor_pos[0]))
        current_nearest_y = min(y_vals, key=lambda y: abs(y - cursor_pos[1]))
        if current_nearest_x != nearest_x or current_nearest_y != nearest_y:
            # print(current_nearest_x, current_nearest_y)
            nearest_x = current_nearest_x
            nearest_y = current_nearest_y
            sample_mat = precomputed_reconstructions[(current_nearest_x, current_nearest_y)]

            note_matrix = sample_mat
            # ax2.imshow(sample_mat, aspect='auto', cmap='Grays', origin='lower')
            im.set_data(sample_mat)
            im.set_clim(vmin=sample_mat.min(), vmax=sample_mat.max())  # <-- Add this line

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
    global weights, latent_vectors, control_points, cursor_pos, precomputed_reconstructions, grid_resolution
    # print('control points: ', control_points)
    # Recompute weights
    start = time.time()
    weights = compute_rbf_weights(control_points, latent_vectors, sigma=RBF_SIGMA)

    # Define the resolution of the 2D space grid
    # grid_resolution = 8  # Higher value means finer grid
    x_vals = np.linspace(-1.5, 1.5, grid_resolution)
    y_vals = np.linspace(-1.5, 1.5, grid_resolution)

    # Initialize a buffer for precomputed reconstructions
    precomputed_reconstructions = {}
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
    z = torch.tensor(latents, dtype=torch.float32)
    # print(z.shape)
    with torch.no_grad():
        samples = model.sample(z, hparams.max_seq_len, 0.1, False)
        print(f'Processing {samples.shape[0]} sequences...')
    for idx, sample in enumerate(samples):
        sample_mat = seq2mat(sample.squeeze(0).tolist(), hparams)
        # width = hparams.width
        # height = hparams.height
        pos_x = x_vals[idx % grid_resolution]
        pos_y = y_vals[idx // grid_resolution]
        # Store the reconstruction in the buffer
        precomputed_reconstructions[(pos_x, pos_y)] = sample_mat
    process_time = time.time()-start
    print(f"Precomputation complete! {process_time:.2f}s")
recalculate_and_update(None)


# Connect button to recalculate_and_update function
def on_button_click(event):
    recalculate_and_update(None)

def reinitialize(event):
    global latent_vectors, max_length, note_matrix, nearest_x, nearest_y
    latent_vectors, max_length = create_latent_vectors(data)
    latent_vectors = np.array(latent_vectors)
    print('\nLatent Vectors:')
    for i, vec in enumerate(latent_vectors):
        print(i, np.round(vec[:5], 2))
    recalculate_and_update(None)
    note_matrix = precomputed_reconstructions[(nearest_x, nearest_y)]

    im.set_data(note_matrix)
    im.set_clim(vmin=note_matrix.min(), vmax=note_matrix.max())
    fig.canvas.draw_idle()

recalculate_button.on_clicked(on_button_click)
reinit_button.on_clicked(reinitialize)

# Initial display
update_latent_display()
plt.show()
stop_event.set()
thread.join()