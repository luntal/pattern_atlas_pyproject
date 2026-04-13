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

##########################
# Load model with metadata
##########################
# ONNX = True
ONNX = False

# Find the latest model in branch-specific models directory
branch_name = safe_branch_name(get_current_branch())
latest_model = latest_in_branch('models', 'model_*.pt')
if latest_model is None:
    raise FileNotFoundError(f"No model files found in models/{branch_name}/")
print(f'Loading model: {latest_model}')

# Load the saved data
checkpoint = torch.load(str(latest_model), map_location='cpu')
model_state_dict = checkpoint['model_state_dict']
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

offset = 0
n_examples = 1
interpolation_steps = 25
data = np.load('data/dataset-merged.npz')['data']
# print(data['tensor'])
print('data shape: ', data.shape)
# maxi = data.shape[0]
maxi = 20
print('using: ', maxi, ' samples')
data = data[0:20]
# print(data)
# sequences = None
syncopation_measures = []
vel_deviations = []
note_densities = []


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
    global syncopation_measures, vel_deviations, note_densities, global_vars
    m_reconstructions, m_originals = calculate(data)
    print('m_reconstructions: ', len(m_reconstructions))
    print('m_originals: ', len(m_originals))
    print('shape: ', m_reconstructions[1].shape)

    current_index = global_vars.get_current_index()
    global_vars.set_matrices(m_reconstructions)
    # matrices = mats
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))  # Three subplots side by side
    plt.subplots_adjust(top=0.96, bottom=0.3, wspace=0.9)  # Adjust top margin to reduce space
    tick_font_size = 5
    label_font_size = 7

    n_timing = hparams.n_timing
    width = hparams.width
    cols = width * n_timing



    original_display = axs[0].imshow(m_originals[current_index], cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    axs[0].invert_yaxis()
    axs[0].set_title(f"Original", fontsize=label_font_size)
    axs[0].set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
    matrix_width = m_originals[0].shape[1]
    axs[0].set_xticks(np.arange(0, matrix_width, 4 * n_timing))  # Set tick positions
    axs[0].set_xticklabels(np.arange(0, matrix_width, 4 * n_timing) // n_timing + 1)

    axs[0].set_yticks(np.arange(-0.5, m_originals[0].shape[0], 1), minor=True)
    axs[0].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[0].tick_params(axis='both', which='minor', labelsize=tick_font_size)



    reconstructions_display = axs[1].imshow(m_reconstructions[current_index], cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
    axs[1].invert_yaxis()
    axs[1].set_title(f"Reconstruction", fontsize=label_font_size)
    axs[1].set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
    matrix_width = m_reconstructions[0].shape[1]
    axs[1].set_xticks(np.arange(0, matrix_width, 4 * n_timing))  # Set tick positions
    axs[1].set_xticklabels(np.arange(0, matrix_width, 4 * n_timing) // n_timing + 1)

    axs[1].set_yticks(np.arange(-0.5, m_reconstructions[0].shape[0], 1), minor=True)
    axs[1].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[1].tick_params(axis='both', which='minor', labelsize=tick_font_size)



    slider_x = 0.15
    slider_width = 1 - 2*slider_x
    slider_height = 0.015



    # Add slider for interpolation index display
    ax_index_slider = plt.axes([slider_x, 0.12, slider_width, slider_height])
    index_slider = Slider(ax_index_slider, 'Index', 0, len(data) - 1, valinit=current_index, valstep=1)

    def update_index(val):
        # global sequences

        _current_index = int(index_slider.val)
        global_vars.set_current_index(_current_index)
        reconstructions_display.set_data(global_vars.get_matrix_at_index(_current_index))
        original_display.set_data(m_originals[_current_index])
        # axs[1].set_title(f"Matrix {_current_index} of {len(m_reconstructions)}", fontsize=label_font_size)
        fig.canvas.draw_idle()


        # c_seq = global_vars.get_sequence_at_index(_current_index)
        # print('token seq: ', c_seq)

        global_vars.midi_processor.seq_index = _current_index



    #
    index_slider.on_changed(update_index)




    # Add button to trigger recalculation

    # Connect button to recalculate_and_update function
    # def on_button_click(event):
    #     recalculate_and_update(None)

    plt.show()


def calculate(sequences):
    global syncopation_measures, vel_deviations, note_densities
    start = time.time()
    data_tensors = torch.tensor(sequences, dtype=torch.long)
    print()

    # print(f'Calculating 2 latent vectors, {interpolation_steps} interpolated vectors and {interpolation_steps + 2} Rhythm sequences!')
    # print(data_tensors.shape)
    # data_tensors[data_tensors >= 60] = 59
    # print('Max value of data_tensors:', torch.max(data_tensors))
    #
    # exit()

    # print(loader)
    latent_space_representations = []
    max_length = 0
    logits, mu, logvar = model(data_tensors)
    z = reparameterize(mu, logvar)

    preds = model.sample(z=z, max_len=hparams.max_seq_len, temperature=0.1, sample=False)
    print(preds.shape)
    sequences = preds.tolist()
    # compute mats with midi processor:
    m_reconstructions = global_vars.generate_matrices(sequences)
    m_originals = global_vars.convert_seq2mat(data)
    return m_reconstructions, m_originals



visualize_with_slider()
# Stop the thread and wait for it to finish before exiting
# global_vars.stop_event.set()
# thread.join()
global_vars.stop_playback()