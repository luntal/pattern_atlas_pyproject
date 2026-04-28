from modules.midi_processor import MidiProcessor
from hparams import hparams
import tkinter as tk
from tkinter import filedialog

def select_folder():
    root = tk.Tk()
    root.withdraw()  # hide main window
    folder_path = filedialog.askdirectory()
    return folder_path

# Example usage
if __name__ == "__main__":
    # abs_path = '/Users/alexanderlunt/Music/Ableton/Projects/Projektrealisation/Drums-Patterns Project/Midi-Clips'
    groove = '/Users/alexanderlunt/Programmierung/datasets/groove'
    maestro_selection_split = '/Users/alexanderlunt/Nextcloud/KI-Labor/Forschung KI-Labor/_Trainingsdaten/MIDI-Files/scores'
    scores = '/Users/alexanderlunt/Nextcloud/KI-Labor/Forschung KI-Labor/_Trainingsdaten/MIDI-Files/scores'
    piano = '/Users/alexanderlunt/Programmierung/datasets/piano'
    lakh = '/Users/alexanderlunt/Nextcloud/KI-Labor/Forschung KI-Labor/_Trainingsdaten/MIDI-Files/LakhMidiDataset-Clean'
    groove_selection = '/Users/alexanderlunt/Music/Ableton/Projects/Global Grooves Project/groove-patterns'
    generic_test = '/Users/alexanderlunt/Nextcloud/KI-Labor/Forschung KI-Labor/_Projekte/Global Grooves/dataset/midi/origial'
    abs_path = groove_selection

    # Initialize processor
    processor = MidiProcessor(hparams)

    tasks = ['select_folder', 'extract', 'only_visualize']
    task = tasks[2]
    dataset_name = 'dataset-groove-selection.npz' 

    if task == 'select_folder':
        folder = select_folder()

        if folder:
            print("Selected folder:", folder)
        else:
            print("No folder selected.")
            exit()

    if task == 'extract':
        # Set drums_only=True when extracting from multi-instrument datasets like Lakh
        processor.extract(midi_folder_path=abs_path, skip=0, take=0, clip=False,
                          randomize_velocity=0, qnt_strength=0, qnt_randomness=0, pitch_randomness=False,
                          drums_only=False, filter_duplicates=True, jump=1, 
                          generic_midi_mapping=False, groove_to_specific=True, generic_to_chromatic=False,
                          bpm_from_filename=True)
        processor.save(dataset_name=dataset_name)

    elif task == 'only_visualize':
        processor.load(dataset_name, take=0, skip=0)
        print('Data shape: ', processor.sequences.shape)
        print('Data size: ', processor.sequences.nbytes / (1024 * 1024), 'MB')

        
        processor.generate_matrices()
        # print(processor.sequences[0])
        processor.visualize(playback=True, print_tokens=True, generic_midi_mapping=True)
