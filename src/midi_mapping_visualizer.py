import tkinter as tk
from tkinter import ttk

# Roland mapping
roland_mapping = {
    36: "Kick", 38: "Snare (Head)", 40: "Snare (Rim)", 37: "Snare X-Stick",
    48: "Tom 1", 50: "Tom 1 (Rim)", 45: "Tom 2", 47: "Tom 2 (Rim)",
    43: "Tom 3 (Head)", 58: "Tom 3 (Rim)", 46: "HH Open (Bow)", 26: "HH Open (Edge)",
    42: "HH Closed (Bow)", 22: "HH Closed (Edge)", 44: "HH Pedal",
    49: "Crash 1 (Bow)", 55: "Crash 1 (Edge)", 57: "Crash 2 (Bow)",
    52: "Crash 2 (Edge)", 51: "Ride (Bow)", 59: "Ride (Edge)", 53: "Ride (Bell)"
}

root = tk.Tk()
root.title("Roland MIDI Mapping (0-127)")

# Canvas + scrollbar
canvas = tk.Canvas(root, width=300, height=400)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
canvas.configure(yscrollcommand=scrollbar.set)
canvas.grid(row=0, column=0, sticky="nsew")
scrollbar.grid(row=0, column=1, sticky="ns")

# Scrollable frame inside canvas
scrollable_frame = ttk.Frame(canvas)
canvas.create_window((0,0), window=scrollable_frame, anchor="nw")

# Update scrollregion
def on_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))
scrollable_frame.bind("<Configure>", on_configure)

# Add header
ttk.Label(scrollable_frame, text="MIDI Note", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
ttk.Label(scrollable_frame, text="Roland Sound", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)

# Add rows 0-127
for i in range(128):
    note_name = roland_mapping.get(i, "")
    
    # Alternate row colors
    bg_color = "#404040" if i % 2 == 0 else "#2a2a2a"

    # Create a frame for the row with background color
    row_frame = tk.Frame(scrollable_frame, bg=bg_color)
    row_frame.grid(row=i + 1, column=0, columnspan=2, sticky="ew")

    # Add labels to the row frame
    tk.Label(row_frame, text=str(i), bg=bg_color).grid(row=0, column=0, padx=5, pady=1, sticky="w")
    tk.Label(row_frame, text=note_name, bg=bg_color).grid(row=0, column=1, padx=5, pady=1, sticky="w")


# --- Add mousewheel / trackpad scrolling support ---
def _on_mousewheel(event):
    # For Mac/Windows
    canvas.yview_scroll(int(-1*(event.delta)), "units")

# Bind events to the canvas
canvas.bind_all("<MouseWheel>", _on_mousewheel)  # MacOS & Windows
canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down

root.mainloop()