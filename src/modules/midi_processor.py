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
import subprocess

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


def remap_drum_pitch_to_24(pitch):
    """
    Remap General MIDI and Roland drum pitches to 24-pitch range (36-59).
    The output range directly corresponds to MIDI notes 36-59 (where 36=kick, 38=snare, 42=hi-hat, etc.)

    This mapping prioritizes Roland drum kit layout and uses free slots for useful GM percussion.

    Target range (MIDI note 36-59):
    36: Kick
    37: Snare X-Stick (Side Stick)
    38: Snare (Head)
    39: Hand Clap
    40: Snare (Rim)
    41: Low Floor Tom
    42: Hi-Hat Closed
    43: Tom 3 (Floor)
    44: Hi-Hat Pedal
    45: Tom 2
    46: Hi-Hat Open
    47: Tom 2 Rim
    48: Tom 1
    49: Crash 1
    50: Tom 1 Rim
    51: Ride (Bow)
    52: Crash 2 Edge (Chinese)
    53: Ride Bell
    54: Tambourine
    55: Crash 1 Edge (Splash)
    56: Cowbell
    57: Crash 2 (Bow)
    58: Tom 3 Rim (Vibraslap)
    59: Ride Edge

    Args:
        pitch (int): Input General MIDI or Roland drum pitch

    Returns:
        int: Remapped pitch in range 36-59, or None if pitch should be excluded
    """
    drum_map = {
        # ROLAND DIRECT MAPPINGS (keep these as-is, they're already in 36-59 range)
        36: 36,  # Kick
        37: 37,  # Snare X-Stick / Side Stick
        38: 38,  # Snare (Head) / Acoustic Snare
        39: 39,  # Hand Clap
        40: 40,  # Snare (Rim) / Electric Snare
        41: 41,  # Low Floor Tom
        42: 42,  # Hi-Hat Closed (Bow)
        43: 43,  # Tom 3 (Head) / High Floor Tom
        44: 44,  # Hi-Hat Pedal
        45: 45,  # Tom 2 / Low Tom
        46: 46,  # Hi-Hat Open (Bow)
        47: 47,  # Tom 2 Rim / Low-Mid Tom
        48: 48,  # Tom 1 / Hi-Mid Tom
        49: 49,  # Crash 1 (Bow)
        50: 50,  # Tom 1 Rim / High Tom
        51: 51,  # Ride (Bow) / Ride Cymbal 1
        52: 52,  # Crash 2 Edge / Chinese Cymbal
        53: 53,  # Ride Bell
        54: 54,  # Tambourine
        55: 55,  # Crash 1 Edge / Splash Cymbal
        56: 56,  # Cowbell
        57: 57,  # Crash 2 (Bow) / Crash Cymbal 2
        58: 58,  # Tom 3 Rim (originally Vibraslap in GM)
        59: 59,  # Ride Edge / Ride Cymbal 2

        # ROLAND NON-STANDARD MAPPINGS (map to equivalent positions)
        22: 42,  # Roland: HH Closed (Edge) -> map to HH Closed
        26: 46,  # Roland: HH Open (Edge) -> map to HH Open

        # GM PITCHES BELOW 36 (map to equivalent slots)
        35: 36,  # GM: Acoustic Bass Drum -> Kick

        # GM PITCHES ABOVE 59 (map to meaningful slots in the 36-59 range)
        60: 48,  # Hi Bongo -> Tom 1 (similar high pitched sound)
        61: 45,  # Low Bongo -> Tom 2 (similar mid pitched sound)
        62: 50,  # Mute Hi Conga -> Tom 1 Rim slot
        63: 47,  # Open Hi Conga -> Tom 2 Rim slot
        64: 41,  # Low Conga -> Low Floor Tom
        65: 48,  # Hi Timbale -> Tom 1
        66: 45,  # Low Timbale -> Tom 2
        67: 56,  # Hi Agogo -> Cowbell (similar metallic sound)
        68: 56,  # Low Agogo -> Cowbell
        69: 54,  # Cabasa -> Tambourine (similar shaker sound)
        70: 54,  # Maracas -> Tambourine
        71: 39,  # Short Whistle -> Hand Clap slot
        72: 39,  # Long Whistle -> Hand Clap slot
        73: 56,  # Short Guiro -> Cowbell (scraping sound, use metallic slot)
        74: 56,  # Long Guiro -> Cowbell
        75: 37,  # Claves -> Side Stick (similar wood click)
        76: 50,  # Hi Wood Block -> Tom 1 Rim slot
        77: 47,  # Low Wood Block -> Tom 2 Rim slot
        78: 58,  # Mute Cuica -> Tom 3 Rim slot (unusual sounds)
        79: 58,  # Open Cuica -> Tom 3 Rim slot
        80: 55,  # Mute Triangle -> Crash 1 Edge (metallic)
        81: 55,  # Open Triangle -> Crash 1 Edge (metallic)
    }

    return drum_map.get(pitch, None)


def get_drum_name_from_24_pitch(pitch):
    """
    Get a descriptive name for MIDI drum pitches in the range 36-59 (Roland mapping).

    Args:
        pitch (int): MIDI pitch in range 36-59

    Returns:
        str: Descriptive name of the drum sound
    """
    # Dictionary mapping MIDI note numbers 36-59 to drum names
    drum_names = {
        36: "Kick",
        37: "Snare X-Stick",
        38: "Snare (Head)",
        39: "Hand Clap",
        40: "Snare (Rim)",
        41: "Low Floor Tom",
        42: "Hi-Hat Closed",
        43: "Tom 3 (Floor)",
        44: "Hi-Hat Pedal",
        45: "Tom 2",
        46: "Hi-Hat Open",
        47: "Tom 2 Rim",
        48: "Tom 1",
        49: "Crash 1",
        50: "Tom 1 Rim",
        51: "Ride (Bow)",
        52: "Chinese Cymbal",
        53: "Ride Bell",
        54: "Tambourine",
        55: "Splash Cymbal",
        56: "Cowbell",
        57: "Crash 2",
        58: "Tom 3 Rim",
        59: "Ride (Edge)",
    }
    return drum_names.get(pitch, f"Unknown ({pitch})")


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


def remap_pitches(notes):
    """
    Remap specific pitches according to a predefined mapping,
    preserving other fields (step_idx, deviation, vel).

    Remapping logic:
    38 -> 39
    36 -> 41
    44 -> 54
    53 -> 59
    59 -> 53

    Args:
        notes (list of dict): each note is {'step_idx': int, 'deviation': int, 'pitch': int, 'vel': int}

    Returns:
        list of dict: same structure as input, only specified pitch values are remapped
    """

    # Define the pitch mapping
    pitch_map = {
        38: 39,
        36: 41,
        44: 54,
        53: 59,
        59: 53
    }

    # Apply the mapping to notes
    remapped_notes = []
    for note in notes:
        new_note = note.copy()
        # Only remap if the pitch is in the mapping, otherwise keep original
        if note['pitch'] in pitch_map:
            new_note['pitch'] = pitch_map[note['pitch']]
        remapped_notes.append(new_note)

    return remapped_notes



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

    def _get_branch_data_folder(self):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        branch_name = 'unknown-branch'
        try:
            result = subprocess.run(
                ['git', '-C', project_root, 'rev-parse', '--abbrev-ref', 'HEAD'],
                check=True,
                capture_output=True,
                text=True,
            )
            branch_name = result.stdout.strip() or 'unknown-branch'
            if branch_name == 'HEAD':
                detached_result = subprocess.run(
                    ['git', '-C', project_root, 'rev-parse', '--short', 'HEAD'],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commit_hash = detached_result.stdout.strip() or 'unknown'
                branch_name = f'detached-{commit_hash}'
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        safe_branch_name = branch_name.replace('/', '_').replace(' ', '_')
        data_folder = os.path.join(project_root, 'data', safe_branch_name)
        os.makedirs(data_folder, exist_ok=True)
        return data_folder


    def load(self, dataset_name, skip=0, take=0):
        if not dataset_name.endswith('.npz'):
            dataset_name += '.npz'

        data_folder = self._get_branch_data_folder()
        path = os.path.join(data_folder, dataset_name)
        npz_data = np.load(path, allow_pickle=True)

        # Support both old and new format
        if 'data' in npz_data:
            sequences_data = npz_data['data']
        elif 'tensor' in npz_data:
            sequences_data = npz_data['tensor']
        else:
            raise ValueError(f"Dataset {dataset_name} does not contain 'data' or 'tensor' key")

        if take <= 0: take = len(sequences_data)
        self.sequences = sequences_data[skip:skip + take]

        # Load metadata if available
        if 'metadata' in npz_data:
            self.metadata = npz_data['metadata'][skip:skip + take].tolist()
        else:
            self.metadata = []
    


    def extract(
            self, midi_folder_path=None, skip=0, take=0, clip=False,
            randomize_velocity=0, qnt_strength=1, qnt_randomness=0, pitch_randomness=False,
            roland=True, drums_only=False, filter_duplicates=True, remap=False, jump=1):
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
        all_filenames = sorted([str(file) for file in Path(abs_path).rglob('*.mid*')])
        print(f'Found {len(all_filenames)} MIDI files in {abs_path}')


        # Filter out duplicate files with numeric suffixes (e.g., .1.mid, .2.mid, .3.mid)
        # Keep only original files without numeric suffixes
        import re
        filenames = []
        duplicate_pattern = re.compile(r'\.\d+\.mid[i]?$')  # Matches .1.mid, .2.midi, .123.mid, etc.

        skipped_duplicates = 0
        for filename in all_filenames:
            if duplicate_pattern.search(filename):
                skipped_duplicates += 1
                continue
            filenames.append(filename)

        n_files = len(filenames)
        print('Number of Files: ', n_files)
        if skipped_duplicates > 0:
            print(f'Skipped {skipped_duplicates} duplicate files with numeric suffixes (.1.mid, .2.mid, etc.)')
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
        skipped_no_drums = 0
        skipped_invalid = 0
        skipped_time_signature = 0
        failed_files = []
        allowed_time_signatures = {(4, 4), (2, 4), (2, 2)}


        for file_idx, filename in enumerate(selection):
            try:
                midi_data = pretty_midi.PrettyMIDI(filename)
            except Exception as e:
                print('-------------------------------------')
                print(f'FAILED: {filename}')
                print(f'Error: {e}')
                skipped_invalid += 1
                failed_files.append((filename, str(e)))
                continue

            print('-------------------------------------')
            print(filename)

            # Only accept files with allowed time signatures.
            # If no explicit signature exists, MIDI convention implies 4/4.
            time_signatures = midi_data.time_signature_changes
            if len(time_signatures) == 0:
                signatures_in_file = {(4, 4)}
            else:
                signatures_in_file = {(ts.numerator, ts.denominator) for ts in time_signatures}

            if not signatures_in_file.issubset(allowed_time_signatures):
                unsupported = sorted(signatures_in_file)
                print(f'Unsupported time signature(s) {unsupported} - skipping file')
                skipped_time_signature += 1
                continue

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

            # Filter for drum tracks if drums_only is enabled
            # this should only be applied on multi-instrument datasets like Lakh, where we want to extract only the drum tracks
            if drums_only:
                drum_instruments = [inst for inst in midi_data.instruments if inst.is_drum]
                if len(drum_instruments) == 0:
                    print('No drum tracks found - skipping file')
                    skipped_no_drums += 1
                    continue
                instruments_to_process = drum_instruments
            else:
                instruments_to_process = midi_data.instruments

            # Guard against empty instrument list
            if len(instruments_to_process) == 0:
                print('No instruments to process - skipping file')
                continue

            for instrument in instruments_to_process:
                # print('instrument name: ', instrument.name)
                width = self.hparams.width
                height = self.hparams.height
                # print(width, height)

                bpm = tempo_changes[0]
                seconds_per_beat = 60 / bpm
                seconds_per_16th_note = seconds_per_beat / 4
                # print(seconds_per_16th_note)
                sorted_midi_notes = sorted(instrument.notes, key=lambda note: note.start)

                # Guard against empty note list
                if len(sorted_midi_notes) == 0:
                    print('No notes in instrument - skipping')
                    continue

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

                    # Check if MIDI note is in valid range (0-127)
                    if pitch < 0 or pitch > 127:
                        print(f'Warning: Invalid MIDI pitch {pitch} - skipping note')
                        continue

                    if clip:
                        # clip notes to range:
                        if pitch < 36 or pitch > 36 + 24:
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

                    # Remap drum pitches when in drums_only mode
                    if drums_only:
                        remapped_pitch = remap_drum_pitch_to_24(pitch)
                        if remapped_pitch is None:
                            # Skip notes that don't map to our drum kit
                            continue
                        pitch = remapped_pitch
                        if pitch < 36 or pitch > 36 + 24:
                            print(f'Warning: MIDI pitch {pitch} out of expected range after remapping - skipping note')
                            exit()
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
                if remap:
                    note_list = remap_pitches(note_list, self.hparams.n_pitch)

                # Check if first bar is empty and remove it if so
                if note_list:
                    first_bar_length = 16  # 16th notes in one bar
                    has_notes_in_first_bar = any(note['step_idx'] < first_bar_length for note in note_list)
                    if not has_notes_in_first_bar:
                        # Shift all notes back by one bar
                        note_list = [
                            {**note, 'step_idx': note['step_idx'] - first_bar_length}
                            for note in note_list
                        ]

                    # Skip what is now the first bar if it has fewer than 6 notes (originally the second bar)
                    notes_in_first_bar = sum(1 for note in note_list if note['step_idx'] < first_bar_length)
                    if notes_in_first_bar < 6:
                        # Remove notes in first bar and shift everything after
                        note_list = [
                            {**note, 'step_idx': note['step_idx'] - first_bar_length}
                            for note in note_list
                            if note['step_idx'] >= first_bar_length
                        ]

                # split the whole seq into sequences of 2 bars:
                len_2bars = 2 * 16  # amount of 16th notes in 2 bars
                qnt_notes_2bars = []
                window = []
                # handle the case of 2bars midi files, which have no 32nd step
                if note_list and note_list[-1]['step_idx'] < len_2bars:
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
                # Apply jump filter: keep only every jump-th window
                filtered_windows = qnt_notes_2bars[::jump]
                # tokenize the 2 bar windows
                ##############################################
                token_list = []
                pattern_index = 0
                for window in filtered_windows:
                    token_window = []
                    prev_idx = 0  # Reset prev_idx for each window to avoid skipping notes
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

        # Filter similar sequences if requested
        sequences_before_filter = len(self.sequences)
        if filter_duplicates and len(self.sequences) > 0:
            print('\nFiltering similar sequences...')
            unique_sequences = []
            unique_metadata = []
            seen_sequences = set()

            for idx, seq in enumerate(self.sequences):
                # Convert list to tuple so it can be hashed
                seq_tuple = tuple(seq)
                if seq_tuple not in seen_sequences:
                    seen_sequences.add(seq_tuple)
                    unique_sequences.append(seq)
                    # Keep metadata at the same index
                    if idx < len(self.metadata):
                        unique_metadata.append(self.metadata[idx])

            self.sequences = unique_sequences
            self.metadata = unique_metadata
            filtered_count = sequences_before_filter - len(self.sequences)
            print(f'Filtered out {filtered_count} duplicate sequences')

        # Print summary statistics
        print('\n' + '=' * 50)
        print(f'EXTRACTION SUMMARY')
        if drums_only:
            print(f'Mode: drums_only')
        if filter_duplicates:
            print(f'Filter duplicates: enabled')
        print(f'Files in selection: {len(selection)}')
        print(f'Files skipped (invalid/corrupt): {skipped_invalid}')
        print(f'Files skipped (time signature): {skipped_time_signature}')
        if drums_only:
            print(f'Files skipped (no drums): {skipped_no_drums}')
        print(f'Total sequences extracted: {sequences_before_filter}')
        if filter_duplicates:
            print(f'Sequences after filtering: {len(self.sequences)}')
            print(f'Duplicates removed: {sequences_before_filter - len(self.sequences)}')
        else:
            print(f'Final sequences: {len(self.sequences)}')

        if failed_files:
            print(f'\nFailed files ({len(failed_files)}):')
            for failed_file, error in failed_files:
                print(f'  - {failed_file}')
                print(f'    Error: {error}')
        print('=' * 50)


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


        data_folder = self._get_branch_data_folder()
        dataset_path = os.path.join(data_folder, dataset_name)

        # Convert metadata to numpy array for saving
        metadata_array = np.array(self.metadata, dtype=object)

        np.savez(dataset_path, data=padded_sequences, metadata=metadata_array)
        print(f"Dataset saved to {dataset_path}")
        print(f"  - Data shape: {padded_sequences.shape}")
        print(f"  - Metadata entries: {len(metadata_array)}")


    def generate_matrices(self, print_to_terminal=False):
        """
        Turn sequences into matrices
        resolution: (steps * n_timing) x pitch
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

        # Count pitch occurrences in all sequences
        pitch_counts = {}
        for seq in self.sequences:
            for token in seq:
                if self.hparams.min_pitch_token <= token < self.hparams.max_pitch_token:
                    pitch = token - self.hparams.min_pitch_token
                    pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

        # Ensure all pitches have an entry, even if count is 0
        for pitch in range(self.hparams.max_pitch_token - self.hparams.min_pitch_token):
            if pitch not in pitch_counts:
                pitch_counts[pitch] = 0

        # Visualize matrices
        mats = self.matrices
        x_positions = np.array([0, 4, 8, 12, 16, 20, 24, 28]) * n_timing
        width = self.hparams.width
        cols = width * n_timing
        beat = 4 * n_timing

        # Create figure with subplots: main matrix and pitch histogram
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)
        ax = fig.add_subplot(gs[0])
        ax_hist = fig.add_subplot(gs[1])

        plt.subplots_adjust(bottom=0.25)
        current_matrix = 0
        # self.matrix = mats[current_matrix]
        im = ax.imshow(mats[current_matrix], cmap='Grays', interpolation='nearest', aspect='auto', vmin=0, vmax=1)
        ax.invert_yaxis()

        # Helper function to format title with metadata
        def get_title(idx):
            title = f"Matrix {idx + 1} of {len(mats)}"
            if idx < len(self.metadata):
                filename, bar_index = self.metadata[idx]
                # Extract just the filename without path
                import os
                basename = os.path.basename(filename)
                title += f"\n{basename} - Bar {bar_index * 2}"
            return title

        ax.set_title(get_title(current_matrix))
        ax.set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
        matrix_width = mats[0].shape[1]
        ax.set_xticks(np.arange(0, matrix_width, 4*n_timing))  # Set tick positions
        ax.set_xticklabels(np.arange(0, matrix_width, 4*n_timing)//n_timing+1)

        pitches = sorted(pitch_counts.keys())
        # Convert pitch indices to MIDI note numbers for labels (pitch range 0-23 -> MIDI 36-59)
        midi_notes = [p + 36 for p in pitches]
        drum_names = [get_drum_name_from_24_pitch(mn) for mn in midi_notes]

        ax.set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
        ax.set_yticks(np.arange(0, len(pitches)))
        ax.set_yticklabels([f"{mn}: {dn}" for mn, dn in zip(midi_notes, drum_names)], fontsize=8)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        # plt.colorbar(im, ax=ax, label='Value (0=Black, 1=White)')

        # Create pitch histogram
        counts = [pitch_counts[p] for p in pitches]


        # Create horizontal bar chart
        ax_hist.barh(pitches, counts, color='steelblue', alpha=0.7)
        ax_hist.set_yticks(pitches)
        ax_hist.set_yticklabels([f"{mn}: {dn}" for mn, dn in zip(midi_notes, drum_names)], fontsize=8)
        ax_hist.set_xlabel('Count', fontsize=10)
        ax_hist.set_ylabel('Pitch (MIDI Note)', fontsize=10)
        ax_hist.set_title('Pitch Occurrences\nin Dataset', fontsize=10)
        ax_hist.grid(axis='x', alpha=0.3)
        # Do not invert y-axis, keep natural order (low pitches at bottom)

        # Add count labels on bars
        for i, (pitch, count) in enumerate(zip(pitches, counts)):
            ax_hist.text(count, pitch, f' {count}', va='center', fontsize=7)

        # Add playback position line (initially hidden at x=-1)
        playback_line = ax.axvline(x=0, color='red', linewidth=1.5, alpha=0.5, linestyle='-')

        # Slider for matrix scrolling
        ax_slider = plt.axes([0.25, 0.1, 0.55, 0.03])
        slider = Slider(ax_slider, 'Matrix', 0, len(mats) - 1, valinit=0, valstep=1)

        # Add previous/next buttons
        from matplotlib.widgets import Button
        ax_prev = plt.axes([0.05, 0.1, 0.09, 0.04])
        ax_next = plt.axes([0.87, 0.1, 0.09, 0.04])
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        # Update function
        def update(val):
            idx = int(slider.val)
            im.set_data(mats[idx])
            # self.matrix = mats[idx]
            self.seq_index = idx
            if print_tokens:
                tokens = [token for token in self.sequences[idx] if  token != self.hparams.pad_token]
                # print('seq: ', tokens)


            ax.set_title(get_title(idx))
            fig.canvas.draw_idle()

        # Button callback functions
        def previous_matrix(event):
            current_val = int(slider.val)
            if current_val > 0:
                slider.set_val(current_val - 1)

        def next_matrix(event):
            current_val = int(slider.val)
            if current_val < len(mats) - 1:
                slider.set_val(current_val + 1)

        # Timer function to update playback position line
        def update_playback_line():
            if playback:
                # Convert playback_position to x coordinate (accounting for n_timing)
                x_pos = self.playback_position * n_timing
                playback_line.set_xdata([x_pos, x_pos])
                fig.canvas.draw_idle()
            # Schedule next update
            return True

        if playback:
            self.playback(fig, update_playback_line)
            # Set up timer to update playback line every 50ms
            timer = fig.canvas.new_timer(interval=50)
            timer.add_callback(update_playback_line)
            timer.start()

        slider.on_changed(update)
        btn_prev.on_clicked(previous_matrix)
        btn_next.on_clicked(next_matrix)
        plt.show()
        if self.stop_event is not None:
            self.stop_event.set()
            self.thread.join()

    def stop_playback(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.thread.join()

    def playback(self, fig=None, update_callback=None):
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
                if counter > self.playback_position:
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

                if counter == self.playback_position:
                    if self.hparams.min_pitch_token <= token < self.hparams.max_pitch_token:
                        pitch = token - self.hparams.min_pitch_token + 36
                        # output_port.send(mido.Message('note_on', note=pitch , velocity=int(vel)))
                        # play_single_note(pitch, int(vel), delay)

                        # print(self.playback_position, pitch, int(vel), delay)
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



            self.playback_position += 1
            self.playback_position %= self.hparams.width
