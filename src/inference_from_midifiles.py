import numpy as np
import torch
# from hparams import hparams
from modules.model import TransformerVAE, reparameterize
from modules.interpolation import interpolate_latent_vectors_spherical
from modules.utilities import seq2mat
from modules.midi_processor import MidiProcessor
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import mido
import threading
from types import SimpleNamespace
import time
from mido import Message, MidiFile, MidiTrack
"""
spherical interpolation between 2 latent vectors.
using midi files from specified folder as input data.
"""

##########################
# Load model with metadata
##########################

# Load the saved data
checkpoint = torch.load('model_256.pt')
# checkpoint = torch.load('model.pt')
model_state_dict = checkpoint['model_state_dict']
hparams_dict = checkpoint['hparams']

# Recreate the hparams object
hparams = SimpleNamespace(**hparams_dict)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = TransformerVAE(hparams)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
print('--' * 10)
print('model config: ')
print('model dimensions: ', hparams.d_model)
print('n_layers: ', hparams.n_layers)
print('n_heads: ', hparams.n_heads)
print('--' * 10)


###########################
# Load Midi Data
###########################
# abs_path = '/Users/alexanderlunt/Music/Ableton/Projects/Projektrealisation/Drums-Patterns Project/Midi-Clips'
abs_path = '/Users/alexanderlunt/Music/Ableton/Projects/Projektrealisation/Drums-Patterns Project/percussion1'
processor = MidiProcessor(hparams)
load = True
# load = False
midi_clips_data_name = 'midi_clips_data.npz'
if load:
    # load precomputed sequences from npz file
    processor.load(midi_clips_data_name)
    data = processor.sequences
else:
    # extract notes from midi files
    processor.extract(abs_path, 0, 0)
    # processor.visualize(playback=False)
    data = processor.convert_to_numpy()
    processor.save(save_path=midi_clips_data_name)
    # exit()


# position = 250
# data = np.load('dataset.npz')['tensor']
# print(data['tensor'])
print('data shape: ', data.shape)
maxi = data.shape[0]
# maxi = hparams.n_samples + 200
print('using: ', maxi, ' samples')


###########################
# Parameters
###########################

seq1 = 0
seq2 = maxi-1
offset = 0
# n_examples = 1
interpolation_steps = 25

playback_position = 0
bpm = 100
beat_length = 60 / bpm
sixteenth_note_length = beat_length / 4
midi_output_port_name = mido.get_output_names()[0]
print('using midi port ', midi_output_port_name)
output_port = mido.open_output(midi_output_port_name, callback=lambda msg: print(msg))
matrices = np.zeros((1, hparams.height, hparams.width))
current_index = 0
playback_cursor_matrix = np.zeros((hparams.height, hparams.width))
playback_cursor_matrix[hparams.height-1, playback_position] = 1


stop_event = threading.Event()


def play_midi():
    global playback_position, matrices, current_index, playback_cursor_matrix

    vel = 0
    while not stop_event.is_set():
        matrix = matrices[current_index]
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
        playback_cursor_matrix = np.roll(playback_cursor_matrix, -1, axis=1)
        playback_position += 1
        playback_position %= hparams.width


stop_event.clear()  # Ensure the event is not set before starting
thread = threading.Thread(target=play_midi)
thread.start()


def visualize_with_slider():
    global matrices, current_index
    mats, seq1_matrix, seq2_matrix = calculate(seq1+ offset, seq2 + offset, interpolation_steps=25)

    matrices = mats
    fig, axs = plt.subplots(3, 1, figsize=(5, 8))  # Three subplots side by side
    plt.subplots_adjust(top=0.96, bottom=0.23, wspace=0.9)  # Adjust top margin to reduce space
    tick_font_size = 5
    label_font_size = 7


    # Static plots for seq1_matrix and seq2_matrix
    axs[0].imshow(seq1_matrix, aspect='auto', cmap='Grays', origin='lower')
    axs[0].set_title("Original Sequence at Index 0", fontsize=label_font_size)
    axs[0].set_xticks(np.arange(-0.5, seq1_matrix.shape[1], 1), minor=True)
    axs[0].set_xticks(np.arange(0, hparams.width, 4))
    axs[0].set_yticks(np.arange(-0.5, seq1_matrix.shape[0], 1), minor=True)
    axs[0].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[0].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[0].tick_params(axis='both', which='minor', labelsize=tick_font_size)


    axs[1].imshow(seq2_matrix, aspect='auto', cmap='Grays', origin='lower')
    axs[1].set_title(f"Original Sequence at Index {len(mats)-1}", fontsize=label_font_size)
    axs[1].set_xticks(np.arange(-0.5, seq2_matrix.shape[1], 1), minor=True)
    axs[1].set_xticks(np.arange(0, hparams.width, 4))
    axs[1].set_yticks(np.arange(-0.5, seq2_matrix.shape[0], 1), minor=True)
    axs[1].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[1].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[1].tick_params(axis='both', which='minor', labelsize=tick_font_size)

    # Dynamic plot with slider
    mat_display = axs[2].imshow(mats[0], aspect='auto', cmap='Grays', origin='lower')
    axs[2].set_title("Reconstructed Sequence with Interpolations", fontsize=label_font_size)
    axs[2].set_xticks(np.arange(-0.5, mats[0].shape[1], 1), minor=True)
    axs[2].set_xticks(np.arange(0, hparams.width, 4))
    axs[2].set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
    axs[2].grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    axs[2].tick_params(axis='both', which='major', labelsize=tick_font_size)
    axs[2].tick_params(axis='both', which='minor', labelsize=tick_font_size)
    
    # Add sliders for seq1 and seq2 selection
    ax_seq1_slider = plt.axes([0.1, 0.06, 0.8, 0.03])
    ax_seq2_slider = plt.axes([0.1, 0.02, 0.8, 0.03])

    seq1_slider = Slider(ax_seq1_slider, 'Seq1', 0, maxi - 1, valinit=seq1, valstep=1)
    seq2_slider = Slider(ax_seq2_slider, 'Seq2', 0, maxi - 1, valinit=seq2, valstep=1)

    ax_button = plt.axes([0.1, 0.15, 0.14, 0.03])  # Position for button
    recalculate_button = Button(ax_button, 'Recalculate')
    recalculate_button.label.set_fontsize(6)

    ax_button_midi = plt.axes([0.25, 0.15, 0.14, 0.03])
    midi_button = Button(ax_button_midi, 'create 1 midi')
    midi_button.label.set_fontsize(6)
    ax_button_midi2 = plt.axes([0.4, 0.15, 0.14, 0.03])
    midi_button2 = Button(ax_button_midi2, 'multi midi')
    midi_button2.label.set_fontsize(6)

    # Add slider for interpolation index display
    ax_index_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    index_slider = Slider(ax_index_slider, 'Index', 0, len(mats) - 1, valinit=current_index, valstep=1)

    # Dynamically update based on slider changes
    def recalculate_and_update(val):
        nonlocal mats, seq1_matrix, seq2_matrix
        global matrices, current_index
        new_seq1 = int(seq1_slider.val)
        new_seq2 = int(seq2_slider.val)
        mats, seq1_matrix, seq2_matrix = calculate(new_seq1 + offset, new_seq2 + offset, interpolation_steps)
        matrices = mats
        current_index = int(index_slider.val)
        axs[0].imshow(seq1_matrix, aspect='auto', cmap='Grays', origin='lower')
        axs[1].imshow(seq2_matrix, aspect='auto', cmap='Grays', origin='lower')
        mat_display.set_data(mats[current_index])

        fig.canvas.draw_idle()

    def update_index(val):
        global current_index, sequences
        current_index = int(index_slider.val)
        mat_display.set_data(mats[current_index])
        fig.canvas.draw_idle()
        # print(sequences[current_index])



    # seq1_slider.on_changed(recalculate_and_update)
    # seq2_slider.on_changed(recalculate_and_update)
    index_slider.on_changed(update_index)
    # Add button to trigger recalculation

    # Connect button to recalculate_and_update function
    def on_button_click(event):
        recalculate_and_update(None)

    recalculate_button.on_clicked(on_button_click)

    def on_midi_click1(event):
        global matrices, current_index
        # # 1. Create the file and a track
        # mid = MidiFile()
        # track = MidiTrack()
        # mid.tracks.append(track)
        # track.append(mido.MetaMessage('set_tempo', tempo=500000)) # 500000 microseconds per beat = 120 bpm
        matrix_to_midi(matrices[current_index], bpm=120)

        print('created single midi rhythm')
    midi_button.on_clicked(on_midi_click1)

    def on_midi_click_multi(event):
        global matrices
        export_all_matrices_to_midi(matrices, bpm=120)
        print('created multi midi rhythm')

    midi_button2.on_clicked(on_midi_click_multi)

    plt.show()

sequences = None
def calculate(seq1, seq2, interpolation_steps=25):
    global sequences, data, device
    start = time.time()
    init = np.array([data[seq1], data[seq2]])
    data_tensors = torch.tensor(init, dtype=torch.long, device=device)
    print()
    print(f'Calculating 2 latent vectors, {interpolation_steps} interpolated vectors. In total {interpolation_steps + 2} Rhythm sequences!')
    # print(data_tensors.shape)
    data_tensors[data_tensors >= 60] = 59
    # print('Max value of data_tensors:', torch.max(data_tensors))
    #
    # exit()

    # print(loader)
    latent_space_representations = []
    max_length = 0



    for seq in data_tensors:
        model_input = seq[seq != 0].view(1, -1) # remove zeros and reshape
        logits, mu, logvar = model(model_input)
        z = reparameterize(mu, logvar)
        latent_space_representations.append(z)
        max_length = max(max_length, z.size(1)) # track maximum seq length


    # interpolate
    interpolated_vectors = interpolate_latent_vectors_spherical(latent_space_representations, steps_between=interpolation_steps)

    # inference
    stack = torch.stack(interpolated_vectors, dim=1).squeeze(0)
    preds = model.sample(z=stack, max_len=hparams.max_seq_len, temperature=0.1, sample=False, device=device)

    samples = preds.tolist()




    # sequences = data[seq1].tolist() + samples + data[seq2].tolist()
    sequences = samples

    # print(f"Original vectors: {len(latent_space_representations)}")
    # print(f"After interpolation: {len(interpolated_vectors)}")


    # samples = []
    # # for z in latent_space_representations:
    # for z in interpolated_vectors:
    #     # print(z[0, :10]*10)
    #     samp = model.sample(z=z, max_len=hparams.max_length, temperature=0.1, sample=False)
    #     samples.append(samp.squeeze(0).tolist())




    mats = []
    for sample in samples:
        sample_mat = seq2mat(sample, hparams)
        mats.append(sample_mat)

    seq1_matrix = seq2mat(data_tensors[0].tolist(), hparams)
    seq2_matrix = seq2mat(data_tensors[1].tolist(), hparams)

    # mats = [seq1_matrix] + mats + [seq2_matrix]

    processing_time = time.time() - start
    print(f"Processing time: {processing_time:.2f} seconds")
    return mats, seq1_matrix, seq2_matrix




    # visualize_with_slider(mats, seq1_matrix, seq2_matrix)


def matrix_to_midi(matrix, bpm=120, ticks_per_beat=480):
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    ticks_per_16th = ticks_per_beat // 4

    # Track how much time we need to wait since the last Note OFF event
    pending_wait_time = 0

    for t in range(matrix.shape[1]):  # Iterate through 16th note columns
        column_notes = np.where(matrix[:, t] > 0)[0]

        if len(column_notes) == 0:
            # If column is empty, accumulate wait time for the next note
            pending_wait_time += ticks_per_16th
            continue

        # Process all simultaneous Note ONs
        for i, p in enumerate(column_notes):
            velocity = int(matrix[p, t] * 127)
            pitch = p + 36  # Adjust to your model's base pitch

            # Only the first note in the chord/column applies the wait time
            wait = pending_wait_time if i == 0 else 0
            track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=wait))
            pending_wait_time = 0

        # Process all simultaneous Note OFFs after one 16th note duration
        for i, p in enumerate(column_notes):
            pitch = p + 36
            # The first OFF waits for the duration of the 16th note
            wait = ticks_per_16th if i == 0 else 0
            track.append(mido.Message('note_off', note=pitch, velocity=0, time=wait))


    mid.save('reconstructed.mid')


def export_all_matrices_to_midi(matrices_list, output_path='reconstructed_multi.mid', bpm=120, ticks_per_beat=480):
    """
    Concatenates a list of matrices into a single MIDI file.
    """
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    ticks_per_16th = ticks_per_beat // 4
    pending_wait_time = 0

    for matrix in matrices_list:
        # matrix shape is (pitches, time_steps)
        for t in range(matrix.shape[1]):
            column_notes = np.where(matrix[:, t] > 0)[0]

            if len(column_notes) == 0:
                pending_wait_time += ticks_per_16th
                continue

            # Note ONs for this time slice
            for i, p in enumerate(column_notes):
                velocity = int(matrix[p, t] * 127)
                pitch = p + 36

                wait = pending_wait_time if i == 0 else 0
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=wait))
                pending_wait_time = 0

            # Note OFFs for this time slice
            for i, p in enumerate(column_notes):
                pitch = p + 36
                wait = ticks_per_16th if i == 0 else 0
                track.append(mido.Message('note_off', note=pitch, velocity=0, time=wait))

    mid.save(output_path)
    print(f"Saved {len(matrices_list)} rhythm patterns to {output_path}")
visualize_with_slider()
# Stop the thread and wait for it to finish before exiting
stop_event.set()
thread.join()