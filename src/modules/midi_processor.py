# import glob
from pathlib import Path
import pretty_midi
import numpy as np
import os
# from vae_rnn_09.hparams import hparams
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mido
import threading
import time
from types import SimpleNamespace
import random

def get_randomness(value, strength=1, invert=False):
    """
    returns a random value between 0 and input value.
    randomness is normally distributed with value as anchor when invert=False
    otherwise mean = 0
    :param invert:
    :param value:
    :return:
    """
    randomness = abs(np.random.normal(0, 0.4))
    randomness = min(randomness, 1)
    randomness *= strength
    if invert:
        value = value * randomness
    else:
        value -= (value * randomness)
    return value


def randomize_pitches(notes, target_range=24):
    """
    Randomly remap the occurring pitches in a list of notes to values in 0..target_range-1
    without overlap, preserving other fields (step_idx, deviation, vel).

    Args:
        notes (list of dict): each note is {'step_idx': int, 'deviation': int, 'pitch': int, 'vel': int}
        target_range (int): the number of target pitches (default=24)

    Returns:
        list of dict: same structure as input, only pitch values are randomized
    """

    # Get all unique pitches in this file
    unique_pitches = sorted({note['pitch'] for note in notes})
    n_unique = len(unique_pitches)

    if n_unique > target_range:
        raise ValueError(f"Number of unique pitches ({n_unique}) exceeds target range ({target_range})")

    # Sample random, non-overlapping target pitches from 0..target_range-1
    randomized_targets = random.sample(range(target_range), n_unique)

    # Create a mapping: original pitch -> randomized pitch
    pitch_map = dict(zip(unique_pitches, randomized_targets))

    # Apply the mapping to notes
    randomized_notes = []
    for note in notes:
        new_note = note.copy()
        new_note['pitch'] = pitch_map[note['pitch']]
        randomized_notes.append(new_note)

    return randomized_notes

class MidiProcessor:
    def __init__(self, hparams: SimpleNamespace):
        self.hparams = hparams
        self.sequences = []
        self.metadata = []
        self.playback_position = 0
        # self.matrix = np.zeros((hparams.height, hparams.width))
        self.stop_event = None
        self.thread = None
        self.seq_index = 0
        self.matrices = []


    def load(self, dataset_name, skip=0, take=0):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', dataset_name)
        data = np.load(path)
        if take <= 0: take = len(data['tensor'])
        self.sequences = data['tensor'][skip:skip + take]
    


    def extract(
            self, midi_folder_path=None, skip=0, take=0, clip=False,
            randomize_velocity=0, qnt_strength=1, qnt_randomness=0, pitch_randomness=False,
            roland=True):
        """
        Extracts sequences from MIDI files and save as list of tokens
        currently sequences are 2 bars long, 16th notes per bar

        !!no sos token!! sos token is added during training only for decoder input!!

        only eos token, eos token should be appended to the end of encoder sequence during inference!
        """
        qnt_strength = max(min(qnt_strength, 1), 0)
        qnt_randomness = max(min(qnt_randomness, 1), 0)
        if midi_folder_path is None:
            print('No MIDI folder path provided. Exiting...')
            exit()
        abs_path = midi_folder_path
        filenames = sorted([str(file) for file in Path(abs_path).rglob('*.mid*')])
        n_files = len(filenames)
        print('Number of Files: ', n_files)
        if skip < 0: skip = 0
        if skip >= n_files: skip = n_files - 1
        limit = n_files - skip
        if take == 0: take = limit
        if take > limit: take = limit
        print(f'Extracting {take} files from {skip} to {skip + take}')
        selection = filenames[skip:skip + take]
        metadata = []
        prev_n_notes = 0
        prev_last_note = 0
        for file_idx, filename in enumerate(selection):
            midi_data = pretty_midi.PrettyMIDI(filename)
            print('-------------------------------------')
            print(filename)
            tempo_times, tempo_changes = midi_data.get_tempo_changes()
            # print(f"Tempo changes: {list(tempo_changes)} at times: {list(tempo_times)}")
            if len(tempo_changes) == 0:
                print('No tempo changes found!')
                continue
            elif len(tempo_changes) > 1:
                print('Multiple tempo changes found!')
                continue
            # compute slightly random quantization strength for every file:
            qnt_randomness = np.random.normal(0, 0.25)
            qnt_randomness = max(min(qnt_randomness, 1), 0)
            current_qnt_strength = qnt_strength * (1-qnt_randomness)
            # print('c_qnt_strength', current_qnt_strength)


            for instrument in midi_data.instruments:
                # print('instrument name: ', instrument.name)
                width = self.hparams.width
                height = self.hparams.height
                # print(width, height)

                bpm = tempo_changes[0]
                seconds_per_beat = 60 / bpm
                seconds_per_16th_note = seconds_per_beat / 4
                # print(seconds_per_16th_note)
                sorted_midi_notes = sorted(instrument.notes, key=lambda note: note.start)
                last_note_16th_index = int(sorted_midi_notes[-1].start // seconds_per_16th_note)
                len_in_bars = np.ceil(last_note_16th_index / 16)
                offset = False
                # if len_in_bars % 2 == 1:
                #     print('odd bar count. cutting 1st bar!')
                #     offset = True
                #     len_in_bars -= 1
                if len_in_bars < 2:
                    print('Too short sequence!')
                    continue
                print(
                    f'bpm: {bpm:.2f}, seconds_per_beat: {seconds_per_beat:.2f}, seconds_per_16th_note: {seconds_per_16th_note:.2f}')
                print(
                    f"Number of notes: {len(sorted_midi_notes)}, last note at: {sorted_midi_notes[-1].start:.2f}s, 16th index: {last_note_16th_index}, len in bars: {len_in_bars:.2f}")
                rhythm_len = 2  # bars
                n_patterns = max(len_in_bars // rhythm_len, 1)

                # take_n_patterns = max(1, n_patterns_full // (1 + n_patterns_full // 10))
                # pattern_index_step_size = n_patterns_full // take_n_patterns
                # print(f"n_patterns_full: {n_patterns_full}, take_n_patterns: {take_n_patterns}")
                print(f"n_patterns_full: {n_patterns}")
                # pattern_index = 0
                # if file_idx > 20:
                #     exit()
                # continue

                step_token = self.hparams.step_token  # 1 sixteenth
                min_pitch_token = self.hparams.min_pitch_token
                min_vel_token = self.hparams.min_vel_token
                sos = self.hparams.sos_token
                eos = self.hparams.eos_token

                # extract midi notes and store them in a list of notes.
                # note = {'step_idx': int, 'deviation': int, 'pitch': int, 'vel': int}
                note_list = []
                for note in sorted_midi_notes:

                    pitch = note.pitch

                    if clip:
                        # clip notes to range:
                        if note.pitch < 36 or note.pitch > 36 + 24:
                            if roland:
                                if pitch == 22:
                                    pitch = 53
                                elif pitch == 26:
                                    pitch = 56
                                else:
                                    continue
                            else:
                                continue
                    if randomize_velocity > 0:
                        vel = note.velocity / 127
                        deviation = float(np.random.uniform(-randomize_velocity, randomize_velocity))
                        vel += deviation
                        vel = min(max(0, vel), 1)
                        vel *= (self.hparams.n_vel - 1)
                        vel = int(vel)
                    else:

                        vel = int(note.velocity / (127/(self.hparams.n_vel-1)))
                    if vel >= self.hparams.n_vel:
                        print('vel: ', vel)
                        print('velocity range exceeded')
                        exit()

                    # retrieve the notes
                    current_time_idx_16th = round(note.start / seconds_per_16th_note)
                    qnt_time = current_time_idx_16th * seconds_per_16th_note
                    deviation = note.start - qnt_time
                    isDeviationNegative = deviation < 0
                    max_deviation = seconds_per_16th_note / 2
                    if abs(deviation) > max_deviation:
                        # print(f'Exceeding: deviation {deviation} > max_deviation {max_deviation}.')
                        deviation = (deviation / abs(deviation)) * max_deviation
                    deviation = deviation / max_deviation  # normalize to - 1 to 1
                    deviation *= 1-current_qnt_strength
                    # deviation = get_randomness(deviation, qnt_randomness, invert=invert)
                    deviation *= (self.hparams.n_timing - 1) / 2 # scale to n_timing range/2 -> -16...16
                    deviation = int(deviation) + int(self.hparams.n_timing / 2)  # shift to right start point -> 0...33
                    # bar_counter = current_time_idx_16th % self.hparams.width

                    # if bar_counter == 0 and isDeviationNegative:
                    #     deviation_center = self.hparams.n_timing // 2
                    #     deviation =  deviation_center + self.hparams.min_timing_token


                    # wrap the pitches to 24 values with 36 as basis
                    pitch = pitch - 12
                    pitch = pitch % 24

                    # pitch = note.pitch - 36
                    if pitch >= self.hparams.max_pitch_token:
                        print('pitch: ', pitch)
                        exit()
                    new_note = {
                        'step_idx': current_time_idx_16th,
                        'deviation': deviation, # mapped to n_timing range
                        'pitch': pitch, #
                        'vel': vel
                    }
                    # qnt_notes.append((current_time_idx_16th, pitch, vel))
                    note_list.append(new_note)

                # randomize
                if pitch_randomness:
                    note_list = randomize_pitches(note_list, self.hparams.n_pitch)


                # split the whole seq into sequences of 2 bars:
                len_2bars = 2 * 16  # amount of 16th notes in 2 bars
                qnt_notes_2bars = []
                window = []
                # handle the case of 2bars midi files, which have no 32nd step
                if note_list[-1]['step_idx'] < len_2bars:
                    new_note = {'step_idx': len_2bars, 'deviation': 0, 'pitch': 0, 'vel': 0}
                    note_list.append(new_note)

                ##############################################
                # if offset=Ture remove the 1st bar:
                if offset == True:
                    note_list = [note for note in note_list if note["step_idx"] > 16]
                    len_2bars = len_2bars - 1
                # split the note stream into 2bar-windows
                for note in note_list:
                    step_idx = note['step_idx']
                    pitch = note['pitch']
                    vel = note['vel']
                    deviation = note['deviation']
                    # print(step_idx, pitch)
                    # print('if ', step_idx // len_2bars)
                    if step_idx // len_2bars == len(qnt_notes_2bars):
                        # print(step_idx)
                        qnt_notes_2bars.append(window)
                        window = []

                    # if step_idx % len_2bars == 0 and deviation < 0:
                    #     deviation = 0 # avoid shifting notes before bar start!
                    window.append((step_idx % len_2bars, deviation, pitch, vel))
                # print('split: ', qnt_notes_2bars)
                # tokenize the 2 bar windows
                ##############################################
                prev_vel = 0
                prev_deviation = self.hparams.n_timing // 2
                prev_idx = 0
                token_list = []
                pattern_index = 0
                for window in qnt_notes_2bars:
                    token_window = []
                    for step_idx, deviation, pitch, vel in window:
                        steps = step_idx - prev_idx
                        for _ in range(steps):
                            token_window.append(step_token)
                        prev_idx = step_idx

                        # insert deviation/timing token:
                        token_window.append(deviation + self.hparams.min_timing_token)

                        # prev_deviation = deviation
                        # if vel != prev_vel:
                        token_window.append(vel + min_vel_token)
                        # prev_vel = vel
                        token_window.append(pitch + min_pitch_token)
                    token_window.append(eos)
                    if len(token_window) < 15:
                        continue
                    token_list.append(token_window)  # list of token sequences
                    self.metadata.append((filename, pattern_index))
                    pattern_index += 1
                # for i in range(4):
                #     print(token_list[i])
                #     print(qnt_notes_2bars[i])

                self.sequences = self.sequences + token_list
                print('\nTotal Number of Sequences: ', len(self.sequences))


    def convert_to_numpy(self):
        # cut to max len 256

        for i, seq in enumerate(self.sequences):
            if len(seq) > 256:
                self.sequences[i] = seq[:256]

        # pad sequences:
        max_length = max(len(seq) for seq in self.sequences)
        padded_sequences = np.zeros((len(self.sequences), max_length), dtype=int)
        for i, seq in enumerate(self.sequences):
            padded_sequences[i, :len(seq)] = seq
        return padded_sequences


    def save(self, dataset_name='dataset.npz'):
        # pad sequences:
        # max_length = max(len(seq) for seq in self.sequences)
        max_length = self.hparams.max_seq_len
        padded_sequences = np.zeros((len(self.sequences), max_length), dtype=int)
        for i, seq in enumerate(self.sequences):
            if len(seq) > max_length:
                seq = seq[:max_length]
                seq[-1] = self.hparams.eos_token
            padded_sequences[i, :len(seq)] = seq

        # save data:
        if not dataset_name.endswith('.npz'):
            dataset_name += '.npz'


        # Prepend data folder path
        data_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
        os.makedirs(data_folder, exist_ok=True)
        dataset_path = os.path.join(data_folder, dataset_name)

        np.savez(dataset_path, tensor=padded_sequences)
        print(f"Tensor saved to {dataset_path}")


    def generate_matrices(self, print_to_terminal=False):
        """
                Turn sequences into matrices
                """
        min_pitch_token = self.hparams.min_pitch_token
        min_vel_token = self.hparams.min_vel_token
        max_pitch_token = self.hparams.max_pitch_token
        max_vel_token = self.hparams.max_vel_token
        step_token = self.hparams.step_token
        min_timing_token = self.hparams.min_timing_token
        max_timing_token = self.hparams.max_timing_token
        n_timing = self.hparams.n_timing

        # n_timing = 2
        def seq2mat(seq):
            cols = self.hparams.width * n_timing
            mat = np.zeros((self.hparams.height, cols))
            idx = 0
            vel = 0
            timing = 0

            def write_wrapped_segment(row: int, start: int, length: int, value: float) -> None:
                """Write 'length' values into mat[row, :] starting at 'start', wrapping around cols."""
                if length <= 0:
                    return
                length = min(length, cols)  # avoid weirdness if length >= cols
                s = start % cols
                e = s + length
                if e <= cols:
                    mat[row, s:e] = value
                else:
                    first_len = cols - s
                    mat[row, s:cols] = value
                    mat[row, 0:(length - first_len)] = value

            for token in seq:
                if min_vel_token <= token < max_vel_token:
                    vel = token - min_vel_token
                    vel /= (self.hparams.n_vel - 1)
                if min_timing_token <= token < max_timing_token:
                    timing = token - min_timing_token - self.hparams.n_timing // 2
                    # timing /= (self.hparams.n_timing-1)
                if min_pitch_token <= token < max_pitch_token:
                    pitch = token - self.hparams.token_bias
                    start = idx * n_timing + timing
                    if n_timing < 2:
                        length = 1
                    else:
                        length = n_timing // 2
                    write_wrapped_segment(pitch, start, length, vel)
                    # end = start + n_timing // 2
                    # mat[pitch, start:end ] = vel
                if token == step_token:
                    idx += 1
                if idx >= self.hparams.width:
                    break
            return mat

        # if print_to_terminal:
        #     for i, seq in enumerate(self.sequences[:11]):
        #         print(i, seq)
        # turn sequences to matrices
        self.matrices = []
        for sample in self.sequences:
            sample_mat = seq2mat(sample)
            # print(sample_mat)
            # print('---' *4)
            self.matrices.append(sample_mat)
        if print_to_terminal:
            print(f"Generated {len(self.matrices)} matrices, shape: {self.matrices[0].shape}")


    def visualize(self, playback=False, print_tokens=False):
        """
        Turn sequences into matrices and visualize them with matplotlib
        """

        n_timing = self.hparams.n_timing

        # Visualize matrices
        mats = self.matrices
        x_positions = np.array([0, 4, 8, 12, 16, 20, 24, 28]) * n_timing
        width = self.hparams.width
        cols = width * n_timing
        beat = 4 * n_timing
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        current_matrix = 0
        # self.matrix = mats[current_matrix]
        im = ax.imshow(mats[current_matrix], cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax.invert_yaxis()
        ax.set_title(f"Matrix {current_matrix + 1} of {len(mats)}")
        ax.set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
        matrix_width = mats[0].shape[1]
        ax.set_xticks(np.arange(0, matrix_width, 4*n_timing))  # Set tick positions
        ax.set_xticklabels(np.arange(0, matrix_width, 4*n_timing)//n_timing+1)

        ax.set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        # plt.colorbar(im, ax=ax, label='Value (0=Black, 1=White)')

        # Slider for matrix scrolling
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
        slider = Slider(ax_slider, 'Matrix', 0, len(mats) - 1, valinit=0, valstep=1)
        # Update function
        def update(val):
            idx = int(slider.val)
            im.set_data(mats[idx])
            # self.matrix = mats[idx]
            self.seq_index = idx
            if print_tokens:
                tokens = [token for token in self.sequences[idx] if  token != self.hparams.pad_token]
                # print('seq: ', tokens)


            ax.set_title(f"Matrix {idx + 1} of {len(mats)}")
            fig.canvas.draw_idle()
        if playback:
            self.playback()

        slider.on_changed(update)
        plt.show()
        if self.stop_event is not None:
            self.stop_event.set()
            self.thread.join()

    def stop_playback(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.thread.join()

    def playback(self):
        port = mido.get_output_names()[0]
        bpm = 100
        beat_length = 60 / bpm
        sixteenth_note_length = beat_length / 4
        output_port = mido.open_output(port, callback=lambda msg: print(msg))
        self.stop_event = threading.Event()

        self.stop_event.clear()  # Ensure the event is not set before starting
        self.thread = threading.Thread(target=self.play_midi, args=(self.playback_position, self.stop_event, output_port, sixteenth_note_length))
        self.thread.start()

    def play_midi(self, playback_position, stop_event, output_port, sixteenth_note_length):
        def play_single_note(pitch, vel, delay=0):
            time.sleep(delay)
            output_port.send(mido.Message('note_on', note=pitch , velocity=vel))
            time.sleep(sixteenth_note_length)
            output_port.send(mido.Message('note_off', note=pitch))
        deviation_strength = 1

        while not stop_event.is_set():
            if len(self.sequences) == 0:
                time.sleep(0.1)
                continue
            seq = self.sequences[self.seq_index]
            # matrix = np.array(matrix)
            # print(playback_position)
            counter = 0
            vel = 0
            delay = sixteenth_note_length

            for token in seq:
                if counter > playback_position:
                    break
                if token == self.hparams.eos_token or token == 0:
                    break

                if self.hparams.min_vel_token <= token < self.hparams.max_vel_token:
                    vel = (token - self.hparams.min_vel_token) * (127/(self.hparams.n_vel-1))
                if self.hparams.min_timing_token <= token < self.hparams.max_timing_token:
                    deviation = token - self.hparams.min_timing_token
                    deviation = deviation - self.hparams.n_timing // 2 # scale to -16 to 16
                    deviation /= (self.hparams.n_timing // 2) # scale to -1 to 1
                    deviation *= (sixteenth_note_length / 2) * deviation_strength
                    delay = sixteenth_note_length + deviation

                if counter == playback_position:
                    if self.hparams.min_pitch_token <= token < self.hparams.max_pitch_token:
                        pitch = token - self.hparams.min_pitch_token + 36
                        # output_port.send(mido.Message('note_on', note=pitch , velocity=int(vel)))
                        # play_single_note(pitch, int(vel), delay)

                        # print(playback_position, pitch, int(vel), delay)
                        threading.Thread(target=play_single_note, args=(pitch, int(vel), delay)).start()

                if token == self.hparams.step_token:
                    counter += 1





            # for pitch in range(self.hparams.height):
            #     if self.matrix[pitch, playback_position] > 0:
            #         vel = int(self.matrix[pitch, playback_position] * 127)
            #         # print(pitch + 36)
            #         output_port.send(mido.Message('note_on', note=pitch + 36, velocity=vel))


            time.sleep(sixteenth_note_length)


            # for i in range(self.hparams.height):
            #     output_port.send(mido.Message('note_off', note=i + 36, velocity=int(vel)))



            playback_position += 1
            playback_position %= self.hparams.width
