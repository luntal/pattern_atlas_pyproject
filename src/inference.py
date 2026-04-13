import numpy as np
import torch
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize, ONNXWrapper
from modules.interpolation import interpolate_latent_vectors_spherical
from modules.utilities import seq2mat, syncopation_measure_weighted, vel_deviation_measure, note_density, play_midi
from modules.midi_processor import MidiProcessor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import mido
import threading
from types import SimpleNamespace
import math
import time
import glob
import os
from modules.branch_paths import latest_in_branch, safe_branch_name, get_current_branch

"""
spherical interpolation between 2 latent vectors.
"""

############################
# Load model with metadata
############################
# ONNX = True
ONNX = False
# Load the saved data
# Find the latest model in branch-specific models directory
branch_name = safe_branch_name(get_current_branch())
latest_model = latest_in_branch('models', 'model_*.pt')
if latest_model is None:
    raise FileNotFoundError(f"No model files found in models/{branch_name}/")
print(f'Loading model: {latest_model}')

checkpoint = torch.load(str(latest_model), map_location='cpu')

model_state_dict = checkpoint['model_state_dict']
# checkpoint = torch.load('model.pt')
# model_state_dict = checkpoint['model_state_dict']
hparams_dict = checkpoint['hparams']

# Recreate the hparams object
hparams = SimpleNamespace(**hparams_dict)

if ONNX:
    model = ONNXWrapper(hparams)
    model.eval()
else:

    model = TransformerVAE(hparams)
    model.load_state_dict(model_state_dict)
    model.eval()
    print('--' * 10)
    print('model config: ')
    print('model dimensions: ', hparams.d_model)
    print('n_layers: ', hparams.n_layers)
    print('n_heads: ', hparams.n_heads)
    print('--' * 10)




###########################
# Parameters
###########################

# position = 250
seq1 = 0
seq2 = 5
offset = 0
n_examples = 1
interpolation_steps = 20
data = np.load('data/dataset-groove.npz')['data']
# print(data['tensor'])
print('data shape: ', data.shape)
# maxi = data.shape[0]
maxi = hparams.n_samples + 200
print('using: ', maxi, ' samples')


# sequences = None
# syncopation_measures = []
# vel_deviations = []
# note_densities = []


class Globals:
    def __init__(self, _hparams=hparams, bpm=100):
        self.playback_position = 0
        self.current_index = 0
        self.matrices = np.zeros((1, _hparams.height, _hparams.width))
        self.width = _hparams.width
        self.height = _hparams.height

        self.bpm = bpm
        self.beat_length = 60 / self.bpm
        self.sixteenth_note_length = self.beat_length / 4

        # self.stop_event = threading.Event()
        # self.stop_event.clear()  # Ensure the event is not set before starting
        self.sequences = None
        self.midi_processor = MidiProcessor(_hparams)
        self.midi_processor.playback()

        self.mp = MidiProcessor(_hparams)

    def step(self, step=1):
        self.playback_position += step
        self.playback_position %= self.width
    def get_step(self):
        return self.playback_position
    def set_current_index(self, index):
        self.current_index = index
    def get_current_index(self):
        return self.current_index
    def set_matrices(self, all_matrices):
        self.matrices = all_matrices
    def get_matrices(self):
        return self.matrices
    def get_matrix_at_index(self, index=0):
        return self.matrices[index]
    def set_sequences(self, sequences):
        self.sequences = sequences
    def get_sequences(self):
        return self.sequences
    def get_sequence_at_index(self, index):
        return self.sequences[index]

    def generate_matrices(self, sequences=None):
        self.midi_processor.sequences = sequences
        self.midi_processor.generate_matrices()
        return self.midi_processor.matrices
    def stop_playback(self):
        self.midi_processor.stop_playback()

    def convert_seq2mat(self, seq):
        self.mp.sequences = seq
        self.mp.generate_matrices()
        return self.mp.matrices

global_vars = Globals()


# thread = threading.Thread(target=play_midi, args=(global_vars,))
# thread.start()


def visualize_with_slider():
    global global_vars
    mats, seq1_matrix, seq2_matrix = calculate(seq1+ offset, seq2 + offset, interpolation_steps=interpolation_steps)

    current_index = global_vars.get_current_index()
    global_vars.set_matrices(mats)
    # matrices = mats
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))  # Three subplots side by side
    plt.subplots_adjust(top=0.96, bottom=0.3, wspace=0.9)  # Adjust top margin to reduce space
    tick_font_size = 5
    label_font_size = 7

    n_timing = hparams.n_timing
    width = hparams.width
    cols = width * n_timing


    axs[0].imshow(seq1_matrix, cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    axs[0].invert_yaxis()
    axs[0].set_title(f"Original Sequence at Index 0", fontsize=label_font_size)
    axs[0].set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
    matrix_width = mats[0].shape[1]
    axs[0].set_xticks(np.arange(0, matrix_width, 4 * n_timing))  # Set tick positions
    axs[0].set_xticklabels(np.arange(0, matrix_width, 4 * n_timing) // n_timing + 1)

    axs[0].set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
    axs[0].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[0].tick_params(axis='both', which='minor', labelsize=tick_font_size)


    axs[1].imshow(seq2_matrix, cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    axs[1].invert_yaxis()
    axs[1].set_title(f"Original Sequence at Index {len(mats)-1}", fontsize=label_font_size)
    axs[1].set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
    matrix_width = mats[0].shape[1]
    axs[1].set_xticks(np.arange(0, matrix_width, 4 * n_timing))  # Set tick positions
    axs[1].set_xticklabels(np.arange(0, matrix_width, 4 * n_timing) // n_timing + 1)

    axs[1].set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
    axs[1].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[1].tick_params(axis='both', which='minor', labelsize=tick_font_size)



    mat_display = axs[2].imshow(mats[0], cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    axs[2].invert_yaxis()
    axs[2].set_title(f"Matrix 0 of {len(mats)}", fontsize=label_font_size)
    axs[2].set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
    matrix_width = mats[0].shape[1]
    axs[2].set_xticks(np.arange(0, matrix_width, 4 * n_timing))  # Set tick positions
    axs[2].set_xticklabels(np.arange(0, matrix_width, 4 * n_timing) // n_timing + 1)

    axs[2].set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
    axs[2].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[2].tick_params(axis='both', which='minor', labelsize=tick_font_size)

    slider_x = 0.15
    slider_width = 1 - 2*slider_x
    slider_height = 0.015

    # Add sliders for seq1 and seq2 selection
    ax_seq1_slider = plt.axes([slider_x, 0.09, slider_width, slider_height])
    ax_seq2_slider = plt.axes([slider_x, 0.06, slider_width, slider_height])

    seq1_slider = Slider(ax_seq1_slider, 'Seq1', 0, maxi - 1, valinit=seq1, valstep=1)
    seq2_slider = Slider(ax_seq2_slider, 'Seq2', 0, maxi - 1, valinit=seq2, valstep=1)

    ax_button = plt.axes([slider_x, 0.02, 0.14, 0.03])  # Position for button
    recalculate_button = Button(ax_button, 'Recalculate')
    recalculate_button.label.set_fontsize(6)

    # Add slider for interpolation index display
    ax_index_slider = plt.axes([slider_x, 0.12, slider_width, slider_height])
    index_slider = Slider(ax_index_slider, 'Index', 0, len(mats) - 1, valinit=current_index, valstep=1)


    # Dynamically update based on slider changes
    def recalculate_and_update(val):
        nonlocal mats, seq1_matrix, seq2_matrix
        new_seq1 = int(seq1_slider.val)
        new_seq2 = int(seq2_slider.val)
        mats, seq1_matrix, seq2_matrix = calculate(new_seq1 + offset, new_seq2 + offset, interpolation_steps)
        global_vars.set_matrices(mats)
        _current_index = int(index_slider.val)
        global_vars.set_current_index(_current_index)
        axs[0].imshow(seq1_matrix, cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        axs[0].invert_yaxis()
        axs[1].imshow(seq2_matrix, cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        axs[1].invert_yaxis()
        mat_display.set_data(mats[_current_index])

        fig.canvas.draw_idle()

    def update_index(val):
        _current_index = int(index_slider.val)
        global_vars.set_current_index(_current_index)
        mat_display.set_data(global_vars.get_matrix_at_index(_current_index))
        axs[2].set_title(f"Matrix {_current_index} of {len(mats)}", fontsize=label_font_size)
        fig.canvas.draw_idle()

        global_vars.midi_processor.seq_index = _current_index



    # seq1_slider.on_changed(recalculate_and_update)
    # seq2_slider.on_changed(recalculate_and_update)
    index_slider.on_changed(update_index)
    # Add button to trigger recalculation

    # Connect button to recalculate_and_update function
    def on_button_click(event):
        recalculate_and_update(None)

    recalculate_button.on_clicked(on_button_click)
    plt.show()


def calculate(seq1, seq2, interpolation_steps=25):
    start = time.time()
    init = np.array([data[seq1], data[seq2]])
    data_tensors = torch.tensor(init, dtype=torch.long)
    print()
    print(f'Calculating 2 latent vectors, {interpolation_steps} interpolated vectors and {interpolation_steps + 2} Rhythm sequences!')
    # print(data_tensors.shape)
    # data_tensors[data_tensors >= 60] = 59
    # print('Max value of data_tensors:', torch.max(data_tensors))
    #
    # exit()

    # print(loader)
    latent_space_representations = []
    max_length = 0

    logits, mu, logvar = model(data_tensors)
    # z = reparameterize(mu, logvar)
    # use mu (mean vector) for deterministic purposes!
    z = mu

    # for seq in data_tensors:
    #     model_input = seq[seq != 0].view(1, -1) # remove zeros and reshape
    #     logits, mu, logvar = model(model_input)
    #     z = reparameterize(mu, logvar)
    #     latent_space_representations.append(z)
    #     max_length = max(max_length, z.size(1)) # track maximum seq length


    interpolated_vectors = interpolate_latent_vectors_spherical(z, steps_between=interpolation_steps)
    print(interpolated_vectors[0].shape)
    print(len(interpolated_vectors))
    # stack = torch.stack(interpolated_vectors, dim=1).squeeze(0)

    # Make sure interpolation output is stacked as float latent vectors
    stack = torch.stack([vec.squeeze(0) for vec in interpolated_vectors], dim=0).float()
    print('stack shape', stack.shape)

    preds = model.sample(z=stack, max_len=hparams.max_seq_len, temperature=0.1, sample=False)
    sequences = preds.tolist()
    # global_vars.set_sequences(sequences)

    # syncopation_measures = []
    # vel_deviations = []
    # note_densities = []
    # mats = []
    # compute mats with midi processor:
    mats = global_vars.generate_matrices(sequences)
    # for sequence in sequences:
    #     sample_mat = seq2mat(sequence, hparams)
        # mats.append(sample_mat)
        # syncopation_measures.append(syncopation_measure_weighted(sample_mat))
        # vel_deviations.append(vel_deviation_measure(sample_mat))
        # note_densities.append(note_density(sample_mat, 2))


    # seq1_matrix = seq2mat(data_tensors[0].tolist(), hparams)
    original_sequences = [data_tensors[0].tolist(), data_tensors[1].tolist()]
    original_sequences = global_vars.convert_seq2mat(original_sequences)
    seq1_matrix = original_sequences[0]
    # seq2_matrix = seq2mat(data_tensors[1].tolist(), hparams)
    seq2_matrix = original_sequences[1]
    processing_time = time.time() - start
    print(f"Processing time: {processing_time:.2f} seconds")
    # print(f'max syncopation measure: {max(syncopation_measures)}')
    return mats, seq1_matrix, seq2_matrix


visualize_with_slider()
# Stop the thread and wait for it to finish before exiting
# global_vars.stop_event.set()
# thread.join()
global_vars.stop_playback()