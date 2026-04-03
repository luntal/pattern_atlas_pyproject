import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import mido

'''
Midi mapping for kinetic bells project (bayern innovativ)
'''
# MIDI note number mapping:
# C1 = 24, G1 = 31
# Swap them, keep everything else unchanged.
pairs = [
    {
        24: 31, # C1-> G1
        31: 24, # G1 -> C1
    },
    {
        37: 38,
        38: 40,
        39: 42,
        40: 48,
        41: 45,
        42: 51,
    }
]
REMAP_PAIRS = pairs[1]


def remap_midi_file(input_path: Path, output_path: Path, remap_pairs: dict[int, int]) -> None:
    """
    Remap MIDI note numbers in a file and save the result to output_path.
    Preserves timing, tracks, messages, and other content as much as possible.
    """
    mid = mido.MidiFile(input_path)

    for track in mid.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                if msg.note in remap_pairs:
                    msg.note = remap_pairs[msg.note]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)


def remap_folder(input_folder: str, output_folder: str, remap_pairs: dict[int, int]) -> None:
    input_root = Path(input_folder)
    output_root = Path(output_folder)

    midi_files = sorted(list(input_root.rglob("*.mid")) + list(input_root.rglob("*.midi")))

    if not midi_files:
        messagebox.showwarning("No MIDI files", "No .mid or .midi files were found in the selected folder.")
        return

    count = 0
    for midi_path in midi_files:
        relative_path = midi_path.relative_to(input_root)
        out_path = output_root / relative_path
        remap_midi_file(midi_path, out_path, remap_pairs)
        count += 1

    messagebox.showinfo(
        "Done",
        f"Remapped {count} MIDI file(s).\n\nOutput folder:\n{output_root}"
    )


class MidiRemapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MIDI Pitch Remapper")
        self.root.geometry("620x180")

        self.input_folder = ""
        self.output_folder = ""

        self.label = tk.Label(root, text="Select a folder, then remap MIDI pitches.", wraplength=380)
        self.label.pack(pady=12)

        self.select_button = tk.Button(root, text="Select Folder", width=20, command=self.select_folder)
        self.select_button.pack(pady=6)

        self.remap_button = tk.Button(root, text="Remap", width=20, command=self.remap, state="disabled")
        self.remap_button.pack(pady=6)

        self.status = tk.Label(root, text="No folder selected.", fg="gray")
        self.status.pack(pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select input MIDI folder")
        if not folder:
            return

        self.input_folder = folder
        self.output_folder = str(Path(folder).with_name(Path(folder).name + "_remapped"))

        self.status.config(text=f"Input: {self.input_folder}\nOutput: {self.output_folder}", fg="black")
        self.remap_button.config(state="normal")

    def remap(self):
        if not self.input_folder:
            messagebox.showwarning("Missing folder", "Please select a folder first.")
            return

        try:
            remap_folder(self.input_folder, self.output_folder, REMAP_PAIRS)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remap MIDI files:\n{e}")


def main():
    root = tk.Tk()
    app = MidiRemapApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()