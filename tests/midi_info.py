import tkinter as tk
from tkinter import filedialog, scrolledtext
import pretty_midi
import os


class MidiInfoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MIDI File Info")
        self.root.geometry("800x600")

        # Select button
        self.select_btn = tk.Button(
            root,
            text="Select MIDI File",
            command=self.select_file,
            font=("Arial", 12),
            padx=20,
            pady=10
        )
        self.select_btn.pack(pady=20)

        # Info display area
        self.info_text = scrolledtext.ScrolledText(
            root,
            wrap=tk.WORD,
            width=90,
            height=30,
            font=("Courier", 10)
        )
        self.info_text.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

    def select_file(self):
        filepath = filedialog.askopenfilename(
            title="Select MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )

        if filepath:
            self.display_midi_info(filepath)

    def display_midi_info(self, filepath):
        self.info_text.delete(1.0, tk.END)

        try:
            midi_data = pretty_midi.PrettyMIDI(filepath)

            # File info
            info = []
            info.append("=" * 80)
            info.append(f"MIDI FILE: {os.path.basename(filepath)}")
            info.append(f"Full path: {filepath}")
            info.append("=" * 80)
            info.append("")

            # Tempo information
            tempo_times, tempo_changes = midi_data.get_tempo_changes()
            info.append("TEMPO INFORMATION:")
            info.append(f"  Number of tempo changes: {len(tempo_changes)}")
            if len(tempo_changes) > 0:
                for i, (time, bpm) in enumerate(zip(tempo_times, tempo_changes)):
                    info.append(f"  Tempo {i+1}: {bpm:.2f} BPM at time {time:.3f}s")
            else:
                info.append("  No tempo information found")
            info.append("")

            # Time signature
            time_sigs = midi_data.time_signature_changes
            info.append("TIME SIGNATURE:")
            info.append(f"  Number of time signature changes: {len(time_sigs)}")
            if len(time_sigs) > 0:
                for i, ts in enumerate(time_sigs):
                    info.append(f"  Time sig {i+1}: {ts.numerator}/{ts.denominator} at time {ts.time:.3f}s")
            else:
                info.append("  No time signature information found")
            info.append("")

            # Duration
            info.append(f"DURATION: {midi_data.get_end_time():.3f} seconds")
            info.append("")

            # Instruments
            info.append(f"INSTRUMENTS: {len(midi_data.instruments)} total")
            info.append("")

            for i, instrument in enumerate(midi_data.instruments):
                info.append(f"Instrument {i+1}:")
                info.append(f"  Name: {instrument.name}")
                info.append(f"  Program: {instrument.program}")
                info.append(f"  Is drum: {instrument.is_drum}")
                info.append(f"  Number of notes: {len(instrument.notes)}")

                if len(instrument.notes) > 0:
                    # Note range
                    pitches = [note.pitch for note in instrument.notes]
                    info.append(f"  Pitch range: {min(pitches)} - {max(pitches)}")

                    # Timing info
                    first_note = min(instrument.notes, key=lambda n: n.start)
                    last_note = max(instrument.notes, key=lambda n: n.end)
                    info.append(f"  First note starts at: {first_note.start:.3f}s")
                    info.append(f"  Last note ends at: {last_note.end:.3f}s")

                    # If tempo is available, show grid alignment
                    if len(tempo_changes) > 0:
                        bpm = tempo_changes[0]
                        seconds_per_beat = 60 / bpm
                        seconds_per_16th = seconds_per_beat / 4

                        # Check first note alignment
                        first_16th_index = first_note.start / seconds_per_16th
                        first_deviation = first_16th_index - round(first_16th_index)

                        info.append(f"  First note 16th grid position: {first_16th_index:.3f}")
                        info.append(f"  First note deviation from grid: {first_deviation:.3f} (16th units)")

                        # Calculate alignment statistics
                        deviations = []
                        for note in instrument.notes:
                            time_in_16ths = note.start / seconds_per_16th
                            nearest_grid = round(time_in_16ths)
                            deviation = abs(time_in_16ths - nearest_grid)
                            deviations.append(deviation)

                        well_aligned = sum(1 for d in deviations if d < 0.4)
                        alignment_score = well_aligned / len(deviations) * 100

                        info.append(f"  Beat alignment score: {alignment_score:.1f}% of notes well-aligned")
                        info.append(f"  Mean deviation: {sum(deviations)/len(deviations):.3f} (16th units)")
                        info.append(f"  Max deviation: {max(deviations):.3f} (16th units)")

                info.append("")

            # Key signature
            key_sigs = midi_data.key_signature_changes
            if len(key_sigs) > 0:
                info.append(f"KEY SIGNATURES: {len(key_sigs)}")
                for i, ks in enumerate(key_sigs):
                    info.append(f"  Key sig {i+1}: {ks.key_number} at time {ks.time:.3f}s")
                info.append("")

            # Display all info
            self.info_text.insert(1.0, "\n".join(info))

        except Exception as e:
            self.info_text.insert(1.0, f"ERROR reading MIDI file:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MidiInfoApp(root)
    root.mainloop()
