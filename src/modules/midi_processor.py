# import glob
import csv
from pathlib import Path
import pretty_midi
import numpy as np
import os
import subprocess
# from vae_rnn_09.hparams import hparams
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mido
import threading
import time
from types import SimpleNamespace
import random
from dataclasses import dataclass, field
from typing import List, Tuple
import re
from modules import compound
from modules.midi_mappings import (
    chromatic_to_generic,
    generic_to_chromatic as generic_to_chromatic_pitch,
    groove_midi_to_groove_midi_merged,
    groove_merged_to_chromatic_specific,
    fan_out_merged_step,
)


def _load_compressed_generic_chromatic_labels() -> dict:
    """Load chromatic pitch (0..53) -> sound label from compressed generic mapping CSV."""
    csv_path = Path(__file__).resolve().parents[2] / 'static' / 'generic_midi_mapping_compressed.csv'
    labels = {}
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chromatic_str = row.get('Chromatic MIDI Pitch', '').strip()
                sound = row.get('Sound', '').strip()
                if chromatic_str == '' or sound == '':
                    continue
                labels[int(chromatic_str)] = sound
    except Exception:
        # Keep visualization functional even if CSV is missing or malformed.
        return {}
    return labels

@dataclass
class ExtractionStats:
    """Statistics from MIDI extraction operation."""
    files_count: int = 0
    skipped_invalid: int = 0
    skipped_time_signature: int = 0
    skipped_no_drums: int = 0
    sequences_before_filter: int = 0
    failed_files: List[Tuple[str, str]] = field(default_factory=list)
    one_bar_patterns: int = 0
    skipped_too_short: int = 0
    truncated_sequences: int = 0  # windows with more notes than max_seq_len - 1 (cut after time)
    merged_duplicates: int = 0    # notes merged because of the same (step, pitch) in a window
    max_notes_per_window: int = 0


class ExtractionCancelled(Exception):
    """Raised by MidiProcessor.extract when should_stop() returns True."""


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

def note_checks(note, n_vel, clip=False, randomize_velocity=0, drums_only=False, roland=True):
    pitch = note.pitch
    # Check if MIDI note is in valid range (0-127)
    if pitch < 0 or pitch > 127:
        print(f'Warning: Invalid MIDI pitch {pitch} - skipping note')
        return None

    if clip:
        # clip notes to range:
        if pitch < 36 or pitch > 36 + 24:
            if roland:
                if pitch == 22:
                    pitch = 42 # hi-hat closed edge -> hi-hat closed
                elif pitch == 26:
                    pitch = 46 # hi-hat open edge -> hi-hat open
                else:
                    return None
            else:
                return None
    if randomize_velocity > 0:
        vel = note.velocity / 127
        v_deviation = float(np.random.uniform(-randomize_velocity, randomize_velocity))
        vel += v_deviation
        vel = min(max(0, vel), 1)
        vel *= (n_vel - 1)
        vel = int(vel)
    else:

        vel = int(note.velocity / (127/(n_vel-1)))
    if vel >= n_vel:
        print('vel: ', vel)
        print('velocity range exceeded')
        exit()
    
    # Remap drum pitches when in drums_only mode
    if drums_only:
        remapped_pitch = remap_drum_pitch_to_24(pitch)
        if remapped_pitch is None:
            # Skip notes that don't map to our drum kit
            return None
        pitch = remapped_pitch
        if pitch < 36 or pitch > 36 + 24:
            print(f'Warning: MIDI pitch {pitch} out of expected range after remapping - skipping note')
            exit()
    # wrap the pitches to 24 values with 36 as basis
    pitch = pitch - 12
    pitch = pitch % 24

    
    return pitch, vel


def get_current_git_branch_name(repo_root):
    """Return a filesystem-safe current git branch name for the given repo root."""
    try:
        branch = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        branch = "default"

    if not branch:
        branch = "default"
    elif branch == "HEAD":
        branch = "detached_head"

    return branch.replace("/", "_").replace("\\", "_")


class MidiProcessor:
    def __init__(self, hparams: SimpleNamespace):
        self.hparams = hparams
        self.sequences = []
        # self.seq_advanced = []
        self.metadata = []
        self.playback_position = 0
        # self.matrix = np.zeros((hparams.height, hparams.width))
        self.stop_event = None
        self.playback_wakeup_event = None
        self.thread = None
        self.seq_index = 0
        self.matrices = []


    def load(self, dataset_name, skip=0, take=0):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_root = os.path.join(repo_root, 'data')
        branch_name = get_current_git_branch_name(repo_root)
        branch_path = os.path.join(data_root, branch_name, dataset_name)
        legacy_path = os.path.join(data_root, dataset_name)

        if os.path.exists(branch_path):
            path = branch_path
        elif os.path.exists(legacy_path):
            path = legacy_path
        else:
            path = branch_path

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

    def filter_duplicate_sequences(self):
        """
        Remove duplicate sequences and corresponding metadata.
        Modifies self.sequences and self.metadata in place.
        """
        if len(self.sequences) == 0:
            return
        
        sequences_before_filter = len(self.sequences)
        print('\nFiltering similar sequences...')
        unique_sequences = []
        unique_metadata = []
        seen_sequences = set()

        for idx, seq in enumerate(self.sequences):
            # Frame arrays are hashed via their bytes
            seq_key = np.asarray(seq, dtype=np.int16).tobytes()
            if seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                unique_sequences.append(seq)
                # Keep metadata at the same index
                if idx < len(self.metadata):
                    unique_metadata.append(self.metadata[idx])

        self.sequences = unique_sequences
        self.metadata = unique_metadata
        filtered_count = sequences_before_filter - len(self.sequences)
        print(f'Filtered out {filtered_count} duplicate sequences')

    def print_extraction_summary(self, stats: ExtractionStats, drums_only: bool, filter_duplicates: bool):
        """
        Print extraction summary statistics.
        
        Args:
            stats: ExtractionStats dataclass containing extraction statistics
            drums_only: whether drums-only mode was enabled
            filter_duplicates: whether duplicate filtering was enabled
        """
        print('\n' + '=' * 50)
        print(f'EXTRACTION SUMMARY')
        if drums_only:
            print(f'Mode: drums_only')
        if filter_duplicates:
            print(f'Filter duplicates: enabled')
        print(f'Files in selection: {stats.files_count}')
        print(f'Files skipped (invalid/corrupt): {stats.skipped_invalid}')
        print(f'Files skipped (time signature): {stats.skipped_time_signature}')
        print(f'Files skipped (too short): {stats.skipped_too_short}')
        print(f'Files with 1-bar patterns: {stats.one_bar_patterns}')
        print(f'Max notes per window: {stats.max_notes_per_window} (max_seq_len {self.hparams.max_seq_len} frames incl. EOS)')
        print(f'Truncated windows: {stats.truncated_sequences}')
        print(f'Merged duplicate notes (same step and pitch): {stats.merged_duplicates}')
        if drums_only:
            print(f'Files skipped (no drums): {stats.skipped_no_drums}')
        print(f'Total sequences extracted: {stats.sequences_before_filter}')
        if filter_duplicates:
            print(f'Sequences after filtering: {len(self.sequences)}')
            print(f'Duplicates removed: {stats.sequences_before_filter - len(self.sequences)}')
        else:
            print(f'Final sequences: {len(self.sequences)}')

        if stats.failed_files:
            print(f'\nFailed files ({len(stats.failed_files)}):')
            for failed_file, error in stats.failed_files:
                print(f'  - {failed_file}')
                print(f'    Error: {error}')
        print('=' * 50)

    def extract(
            self, midi_folder_path=None, skip=0, take=0, clip=False,
            randomize_velocity=0, qnt_strength=1, qnt_randomness=0, pitch_randomness=False,
            roland=True, drums_only=False, filter_duplicates=True, remap=False, jump=1,
            generic_midi_mapping=False, groove_to_specific=False, generic_to_chromatic=False,
            progress_callback=None, should_stop=None):
        """
        Extracts sequences from MIDI files and save as list of tokens
        currently sequences are 2 bars long, 16th notes per bar

        Optional hooks (used by the server GUI):
          progress_callback(file_idx, n_files, filename, stats, n_sequences) is called before each file
          and once more after the last one (file_idx == n_files).
          should_stop() -> bool is polled before each file; True raises ExtractionCancelled.

        !!no sos token!! sos token is added during training only for decoder input!!

        only eos token, eos token should be appended to the end of encoder sequence during inference!
        """
        if groove_to_specific and generic_midi_mapping:
            warning = (
                "groove_to_specific=True requires generic_midi_mapping=False. "
                "Stop extraction and disable generic_midi_mapping."
            )
            print(f"WARNING: {warning}")
            raise ValueError(warning)

        if generic_to_chromatic and (generic_midi_mapping or groove_to_specific):
            warning = (
                "generic_to_chromatic=True requires generic_midi_mapping=False and groove_to_specific=False. "
                "Stop extraction and disable the conflicting options."
            )
            print(f"WARNING: {warning}")
            raise ValueError(warning)

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
        allowed_time_signatures = {(4, 4), (2, 4), (2, 2)}

        statistics = ExtractionStats()
        statistics.files_count = len(selection)
        self.last_stats = statistics
        for file_idx, filename in enumerate(selection):
            if should_stop is not None and should_stop():
                raise ExtractionCancelled()
            if progress_callback is not None:
                progress_callback(file_idx, len(selection), filename, statistics, len(self.sequences))
            file_variant_index = sum(ord(ch) for ch in Path(filename).name)
            fanout_state = {
                'merged_cursors': {},
                'pitch_usage': {},
            }
            try:
                midi_data = pretty_midi.PrettyMIDI(filename)
            except Exception as e:
                print('-------------------------------------')
                print(f'FAILED: {filename}')
                print(f'Error: {e}')
                statistics.skipped_invalid += 1
                statistics.failed_files.append((filename, str(e)))
                continue

            print('-------------------------------------')
            print(file_idx, filename)

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
                statistics.skipped_time_signature += 1
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
            if drums_only:
                drum_instruments = [inst for inst in midi_data.instruments if inst.is_drum]
                if len(drum_instruments) == 0:
                    print('No drum tracks found - skipping file')
                    statistics.skipped_no_drums += 1
                    continue
                instruments_to_process = drum_instruments
            else:
                instruments_to_process = midi_data.instruments

            # Guard against empty instrument list
            if len(instruments_to_process) == 0:
                print('No instruments to process - skipping file')
                continue

            # check bpm from filename if option enabled
            bpm_from_name = None
            # if bpm_from_filename:
            filename_lower = Path(filename).stem.lower()
            bpm_match = re.search(r'(\d{2,3})\s*bpm$', filename_lower)
            # bpm_match = re.search(r'(\d{2,3})\s?bpm', filename_lower)
            if bpm_match:
                bpm_from_name = int(bpm_match.group(1))
                print(f'BPM from filename: {bpm_from_name}')
                # tempo_changes[0] = bpm_from_name
                
            # else:
            #     print('No BPM found in filename, using MIDI tempo.')
                

            for instrument in instruments_to_process:
                # print('instrument name: ', instrument.name)
                width = self.hparams.width
                height = self.hparams.height

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
                
                if len_in_bars == 1:
                    # Exactly 1 bar: duplicate it into bar 2 to form a valid 2-bar window.
                    bar_offset_seconds = 16 * seconds_per_16th_note
                    duplicated = [
                        pretty_midi.Note(
                            velocity=n.velocity,
                            pitch=n.pitch,
                            start=n.start + bar_offset_seconds,
                            end=n.end + bar_offset_seconds,
                        )
                        for n in sorted_midi_notes
                    ]
                    sorted_midi_notes = sorted_midi_notes + duplicated
                    last_note_16th_index = int(sorted_midi_notes[-1].start // seconds_per_16th_note)
                    len_in_bars = np.ceil(last_note_16th_index / 16)
                    print('1-bar pattern duplicated to 2 bars.')
                    statistics.one_bar_patterns += 1
                elif len_in_bars < 2:
                    print('Too short sequence!')
                    statistics.skipped_too_short += 1
                    continue
                print(
                    f'bpm: {bpm:.2f}, seconds_per_beat: {seconds_per_beat:.2f}, seconds_per_16th_note: {seconds_per_16th_note:.2f}')
                print(
                    f"Number of notes: {len(sorted_midi_notes)}, last note at: {sorted_midi_notes[-1].start:.2f}s, 16th index: {last_note_16th_index}, len in bars: {len_in_bars:.2f}")
                rhythm_len = 2  # bars
                n_patterns = int(max(len_in_bars // rhythm_len, 1))

                print(f"n_patterns_full: {n_patterns}")


                # Compound frames store raw field values (pitch 0..n_pitch-1, velocity 0..n_vel-1,
                # deviation 0..n_timing-1); the per-field offsets are added by modules.compound.
                center_timing = compound.center_timing(self.hparams)
                n_timing = int(self.hparams.n_timing)

                # extract midi notes and store them in a list of notes.
                # note = {'step_idx': int, 'deviation': int, 'pitch': int, 'vel': int}
                note_list = []
                notes_advanced = []
                prev_note_start = 0
                for note in sorted_midi_notes:
                    
                    result = note_checks(note, n_vel=self.hparams.n_vel, clip=clip, randomize_velocity=randomize_velocity, drums_only=drums_only, roland=roland)
                    if result is None:
                        continue
                    pitch, vel = result

                    if generic_midi_mapping or groove_to_specific:
                        # Use raw MIDI pitch before the 0-23 wrapping performed by note_checks.
                        merged_note = groove_midi_to_groove_midi_merged(note.pitch)
                    else:
                        merged_note = None

                    # retrieve the notes
                    window_size_in_16th = self.hparams.width
                    current_time_idx_16th = round(note.start / seconds_per_16th_note)
                    current_16th = note.start / seconds_per_16th_note
                    current_window = current_time_idx_16th // (window_size_in_16th)
                    current_step_in_window = current_time_idx_16th % (window_size_in_16th)
                    
                    
                    qnt_time = current_time_idx_16th * seconds_per_16th_note
                    dev = note.start - qnt_time
                    dev_norm = current_16th - current_time_idx_16th
                    # Normalize microshift to [-1, 1], where +/-1 equals half a sixteenth note.
                    shift_norm = max(-1.0, min(1.0, dev_norm * 2.0))
                    dev_idx = int(round((shift_norm * (n_timing // 2)) + center_timing))
                    dev_idx = max(0, min(n_timing - 1, dev_idx))

                    note_interval = note.start - prev_note_start
                    note_interval_in_16th = note_interval / seconds_per_16th_note
                    first_note_in_window = not any(
                        n['c_window'] == current_window
                        for n in notes_advanced
                    )
                    
                    if first_note_in_window:
                        current_previous_note_start = current_window * window_size_in_16th * seconds_per_16th_note
                        window_note_interval = note.start - current_previous_note_start
                        window_note_interval = max(0, window_note_interval)  # Ensure non-negative
                        window_note_interval_in_16th = window_note_interval / seconds_per_16th_note
                        window_dev_norm = shift_norm
                    else:
                        window_note_interval = None
                        window_note_interval_in_16th = None
                        window_dev_norm = None


                    advanced_note = {
                        'start': note.start,
                        'pitch': note.pitch,
                        'pitch_idx': pitch,
                        'velocity': note.velocity,
                        'velocity_idx': vel,
                        'current_16th': current_16th,
                        'current_time_idx_16th': current_time_idx_16th,
                        'qnt_time': qnt_time,
                        'deviation': dev,
                        'dev_norm': dev_norm,
                        'dev_idx': dev_idx,
                        'c_window': current_window,
                        'c_step_in_window': current_step_in_window,
                        'note_interval': note_interval,
                        'note_interval_in_16th': note_interval_in_16th,
                        'filename': filename,
                        'bpm': bpm,
                        'bpm_from_name': bpm_from_name,
                        'first_note_in_window': first_note_in_window,
                        'window_note_interval': window_note_interval,
                        'window_note_interval_in_16th': window_note_interval_in_16th,
                        'merged_note': merged_note,
                        'window_dev_norm': window_dev_norm
                    }
                    # print(f"Note: start={note.start:.2f}s, pitch={note.pitch}, vel={note.velocity}, current_16th={current_16th:.2f}, ")
                    prev_note_start = note.start
                    notes_advanced.append(advanced_note)

                # Map generic input note IDs directly to chromatic before tokenization.
                if generic_to_chromatic and notes_advanced:
                    for i, n in enumerate(notes_advanced):
                        chromatic_note = generic_to_chromatic_pitch(int(n['pitch']))
                        notes_advanced[i]['pitch_idx'] = int(chromatic_note)


                # Apply feature-based fan-out to assign final chromatic pitches (0-53).
                if generic_midi_mapping and notes_advanced:
                    window_groups: dict[int, dict[int, list[int]]] = {}
                    for i, n in enumerate(notes_advanced):
                        c_window = n['c_window']
                        c_step = n['current_time_idx_16th']
                        window_groups.setdefault(c_window, {}).setdefault(c_step, []).append(i)

                    for c_window in sorted(window_groups.keys()):
                        # Keep merged->chromatic mapping stable within each 2-bar window.
                        window_fixed_map: dict[int, int] = {}
                        step_groups = window_groups[c_window]
                        for step_idx in sorted(step_groups.keys()):
                            step_indices = step_groups[step_idx]
                            step_events = [
                                {'merged_note': notes_advanced[i]['merged_note'],
                                 'velocity': notes_advanced[i]['velocity']}
                                for i in step_indices
                            ]
                            fanned = fan_out_merged_step(
                                step_events,
                                variant_index=file_variant_index,
                                state=fanout_state,
                                fixed_map=window_fixed_map,
                            )
                            for list_idx, fanned_event in zip(step_indices, fanned):
                                notes_advanced[list_idx]['pitch_idx'] = int(fanned_event['chromatic_note'])

                # Deterministic merged->chromatic-specific mapping.
                if groove_to_specific and notes_advanced:
                    for i, n in enumerate(notes_advanced):
                        chromatic_note = groove_merged_to_chromatic_specific(int(n['merged_note']))
                        notes_advanced[i]['pitch_idx'] = int(chromatic_note)

                # self.seq_advanced.append(notes_advanced)
                token_list, metadata = self.notes_advanced_to_token_sequences_stream(
                    notes_advanced,
                    self.hparams.width,
                    2,  # at least one note frame + EOS
                    max_window_exclusive=n_patterns,
                    jump=jump,
                    stats=statistics,
                )
                self.metadata.extend(metadata)


                self.sequences = self.sequences + token_list
                print('\nTotal Number of Sequences: ', len(self.sequences))

        if progress_callback is not None:
            progress_callback(len(selection), len(selection), '', statistics, len(self.sequences))

        # Filter similar sequences if requested
        statistics.sequences_before_filter = len(self.sequences)
        if filter_duplicates:
            self.filter_duplicate_sequences()

        
        self.print_extraction_summary(statistics, drums_only, filter_duplicates)



    def convert_to_numpy(self):
        """Sequences as int16 frame array [N, max_seq_len, 4]."""
        return compound.pad_frames(self.sequences, int(self.hparams.max_seq_len))


    def save(self, dataset_name='dataset.npz'):
        # frames [N, max_seq_len, 4] (int16), cut with EOS / padded with PAD frames
        padded_sequences = self.convert_to_numpy()

        # save data:
        if not dataset_name.endswith('.npz'):
            dataset_name += '.npz'


        # Save by git branch to keep datasets isolated per branch.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_root = os.path.join(repo_root, 'data')
        branch_name = get_current_git_branch_name(repo_root)
        data_folder = os.path.join(data_root, branch_name)
        os.makedirs(data_folder, exist_ok=True)
        dataset_path = os.path.join(data_folder, dataset_name)

        # Convert metadata to numpy array for saving
        metadata_array = np.array(self.metadata, dtype=object)

        import json
        info = dict(compound.format_info(self.hparams), kind='extract', name=dataset_name)
        np.savez(dataset_path, data=padded_sequences, metadata=metadata_array, info=np.array(json.dumps(info)))
        print(f"Dataset saved to {dataset_path}")
        print(f"  - Branch folder: {branch_name}")
        print(f"  - Data shape: {padded_sequences.shape}")
        print(f"  - Metadata entries: {len(metadata_array)}")


    def generate_matrices(self, print_to_terminal=False):
        """
        Turn sequences into matrices
        resolution: (steps * n_timing) x pitch
        """
        def seq2mat(seq):
            return compound.frames_to_matrix(seq, self.hparams).astype(np.float64)

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


    def visualize(self, playback=False, print_tokens=False, generic_midi_mapping=False):
        """
        Turn sequences into matrices and visualize them with matplotlib
        """

        n_timing = self.hparams.n_timing

        # Count pitch occurrences in all sequences
        pitch_counts = {}
        for seq in self.sequences:
            for pitch in compound.frames_to_raw(seq, self.hparams)[:, compound.FIELD_PITCH].tolist():
                pitch_counts[pitch] = pitch_counts.get(pitch, 0) + 1

        # Ensure all pitches have an entry, even if count is 0
        for pitch in range(int(self.hparams.n_pitch)):
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
                filename, bar_index, window_bpm = self.metadata[idx]
                # Extract just the filename without path
                import os
                basename = os.path.basename(filename)
                title += f"\n{basename} - Bar {bar_index * 2} (BPM: {window_bpm})"
            return title

        ax.set_title(get_title(current_matrix))
        ax.set_xticks(np.arange(-0.5, cols, n_timing), minor=True)
        matrix_width = mats[0].shape[1]
        ax.set_xticks(np.arange(0, matrix_width, 4*n_timing))  # Set tick positions
        ax.set_xticklabels(np.arange(0, matrix_width, 4*n_timing)//n_timing+1)

        pitches = sorted(pitch_counts.keys())
        compressed_labels = _load_compressed_generic_chromatic_labels()

        def format_pitch_label(pitch_idx):
            if pitch_idx in compressed_labels:
                return f"{pitch_idx}: {compressed_labels[pitch_idx]}"
            # Fallback for non-compressed 24-pitch setups.
            midi_note = pitch_idx + 36
            return f"{midi_note}: {get_drum_name_from_24_pitch(midi_note)}"

        y_labels = [format_pitch_label(p) for p in pitches]

        ax.set_yticks(np.arange(-0.5, mats[0].shape[0], 1), minor=True)
        ax.set_yticks(np.arange(0, len(pitches)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        # plt.colorbar(im, ax=ax, label='Value (0=Black, 1=White)')

        # Create pitch histogram
        counts = [pitch_counts[p] for p in pitches]


        # Create horizontal bar chart
        ax_hist.barh(pitches, counts, color='steelblue', alpha=0.7)
        ax_hist.set_yticks(pitches)
        ax_hist.set_yticklabels(y_labels, fontsize=8)
        ax_hist.set_xlabel('Count', fontsize=10)
        ax_hist.set_ylabel('Pitch (Chromatic Index)', fontsize=10)
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
            if playback and self.playback_wakeup_event is not None:
                self.playback_wakeup_event.set()
            if print_tokens:
                frames = compound.frames_to_raw(self.sequences[idx], self.hparams)
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
                # Align line with the quantization center used for zero microshift.
                x_pos = self.playback_position * n_timing + n_timing
                playback_line.set_xdata([x_pos, x_pos])
                fig.canvas.draw_idle()
            # Schedule next update
            return True

        if playback:
            self.playback(fig, update_playback_line, generic_midi_mapping=generic_midi_mapping)
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

    def _get_sequence_bpm(self, seq_index, fallback_bpm=100.0):
        """Return BPM for the given sequence index from metadata if available."""
        if seq_index < 0 or seq_index >= len(self.metadata):
            return fallback_bpm

        metadata_item = self.metadata[seq_index]
        if isinstance(metadata_item, (list, tuple, np.ndarray)) and len(metadata_item) >= 3:
            try:
                bpm = float(metadata_item[2])
                if bpm > 0:
                    return bpm
            except (TypeError, ValueError):
                pass

        return fallback_bpm

    def playback(self, fig=None, update_callback=None, generic_midi_mapping=False):
        port = mido.get_output_names()[0]
        output_port = mido.open_output(port, callback=lambda msg: print(msg))
        self.stop_event = threading.Event()
        self.playback_wakeup_event = threading.Event()

        self.stop_event.clear()  # Ensure the event is not set before starting
        self.playback_wakeup_event.clear()
        self.thread = threading.Thread(
            target=self.play_midi,
            args=(
                self.playback_position,
                self.stop_event,
                output_port,
                generic_midi_mapping,
            ),
        )
        self.thread.start()

    def play_midi(self, playback_position, stop_event, output_port, generic_midi_mapping=False):
        def play_single_note(pitch, vel, delay=0, note_length=0.0):
            time.sleep(delay)
            output_port.send(mido.Message('note_on', note=pitch , velocity=vel))
            time.sleep(note_length)
            output_port.send(mido.Message('note_off', note=pitch))
        deviation_strength = 1

        while not stop_event.is_set():
            if len(self.sequences) == 0:
                time.sleep(0.1)
                continue

            current_seq_index = self.seq_index
            bpm = self._get_sequence_bpm(current_seq_index)
            beat_length = 60 / bpm
            sixteenth_note_length = beat_length / 4
            seq = self.sequences[current_seq_index]
            # matrix = np.array(matrix)
            # print(playback_position)
            interval_start = float(self.playback_position)
            interval_end = interval_start + 1.0

            for note in compound.frames_to_notes(seq, self.hparams):
                event_step = note['step'] + (0.5 * note['shift'] * deviation_strength)
                if not (interval_start <= event_step < interval_end):
                    continue

                delay = (event_step - interval_start) * sixteenth_note_length
                delay = max(0.0, delay)

                chromatic_pitch = note['pitch']
                if generic_midi_mapping:
                    pitch = chromatic_to_generic(chromatic_pitch)
                else:
                    pitch = chromatic_pitch + 36
                if pitch < 0 or pitch > 127:
                    continue

                threading.Thread(
                    target=play_single_note,
                    args=(pitch, int(note['vel']), delay, sixteenth_note_length),
                ).start()



            if self.playback_wakeup_event is not None:
                woke_up = self.playback_wakeup_event.wait(timeout=sixteenth_note_length)
                if woke_up:
                    self.playback_wakeup_event.clear()
                    continue
            else:
                time.sleep(sixteenth_note_length)

            self.playback_position += 1
            self.playback_position %= self.hparams.width

    

    def notes_advanced_to_token_sequences_stream(
        self,
        notes_advanced,
        steps_per_window=32,
        min_seq_len=0,
        max_window_exclusive=None,
        jump=1,
        stats=None,
    ):
        """Notes of one instrument -> one compound frame sequence per 2-bar window.

        Each note becomes a frame [step_in_window, pitch, velocity, deviation] (via modules.compound: sorted by
        (step, pitch), duplicates merged, EOS, PAD up to max_seq_len, too long windows cut after time).
        min_seq_len counts frames including EOS.
        """
        sequences = []
        metadata = []
        if not notes_advanced:
            return sequences, metadata

        max_len = int(self.hparams.max_seq_len)
        bpm = notes_advanced[0]["bpm"] if notes_advanced[0]["bpm_from_name"] is None else notes_advanced[0]["bpm_from_name"]

        windows: dict[int, list] = {}
        for n in notes_advanced:
            w = n["c_window"]
            # Keep only full 2-bar windows when requested.
            if max_window_exclusive is not None and w >= max_window_exclusive:
                continue
            # Skip windows that are not multiples of jump.
            if w % jump != 0:
                continue
            windows.setdefault(w, []).append(
                (int(n["c_step_in_window"]) % int(steps_per_window), int(n["pitch_idx"]), int(n["velocity_idx"]), int(n["dev_idx"]))
            )

        for w in sorted(windows):
            notes = windows[w]
            frames, n_notes, truncated = compound.notes_to_frames(notes, self.hparams, max_len=max_len, return_info=True)
            if stats is not None:
                unique = len({(s, p) for s, p, _, _ in notes})
                stats.merged_duplicates += len(notes) - unique
                stats.max_notes_per_window = max(stats.max_notes_per_window, unique)
                stats.truncated_sequences += int(truncated)
            if n_notes + 1 < min_seq_len:
                continue
            sequences.append(frames)
            metadata.append((notes_advanced[0]["filename"], w, bpm))
        return sequences, metadata
