import random



def groove_midi_to_groove_midi_compressed(input_note):
    mapping = {
        22: 42,  # HH Closed (Edge) -> Closed Hi-Hat (42)
        26: 46,  # HH Open (Edge) -> Open Hi-Hat (46)
        36: 36,  # Kick -> Bass (36)
        37: 37,  # Snare X-Stick -> Snare (38)
        38: 38,  # Snare (Head) -> Snare (38)
        40: 40,  # Snare (Rim) -> Snare (38)
        42: 42,  # HH Closed (Bow) -> Closed Hi-Hat (42)
        43: 43,  # Tom 3 (Head) -> High Floor Tom (43)
        44: 44,  # HH Pedal -> Closed Hi-Hat (42)
        45: 45,  # Tom 2 -> Low-Mid Tom (47)
        46: 46,  # HH Open (Bow) -> Open Hi-Hat (46)
        47: 45,  # Tom 2 (Rim) -> Low-Mid Tom (47)
        48: 48,  # Tom 1 -> High Tom (48)
        49: 49,  # Crash 1 (Bow) -> Crash Cymbal (49)
        50: 48,  # Tom 1 (Rim) -> High Tom (48)
        51: 51,   # Ride (Bow) -> Ride Cymbal (51)
        52: 49,   # Crash 2 (Edge) -> Crash Cymbal (49)
        53: 51,   # Ride (Bell) -> Ride Cymbal (51)
        55: 49,   # Crash 1 (Edge) -> Crash Cymbal (49)
        57: 49,   # Crash 2 (Bow) -> Crash Cymbal (49)
        58:43,    # Tom3(Rim) -> High Floor Tom(43)
        59:51,    # Ride (Edge) -> Ride Cymbal (51)
    }
    return mapping.get(input_note, input_note)
'''
Compressed mapping:
Kick -> 36
Snare X Stick -> 37
Snare (Head) -> 38
Snare (Rim) -> 40
HH Closed (Bow) -> 42
Tom 3 (Head) -> 43
HH Pedal -> 44
Tom 2 -> 45
HH Open (Bow) -> 46
Tom 1 -> 48
Crash 1 (Bow) -> 49
Tom 1 (Rim) -> 50
Ride (Bow) -> 51
'''



def groove_midi_compressed_to_groove_midi_compressed_ZEROED(input_note):
    mapping ={
        36: 0,
        37: 1,
        38: 2,
        40: 3,
        42: 4,
        43: 5,
        44: 6,
        45: 7,
        46: 8,
        48: 9,
        49: 10,
        50: 11,
        51: 12,
        53: 13,
        55: 14,
    }
    return mapping.get(input_note, input_note)

generic_to_chromatic_mapping = {
    0: 0,    # C-2 -> C-2
    1: 1,    # C#-2 -> C#-2
    2: 2,    # D-2 -> D-2
    3: 3,    # Eb-2 -> Eb-2
    4: 4,    # E-2 -> E-2
    5: 5,    # F-2 -> F-2
    6: 6,    # F#-2 -> F#-2
    7: 7,    # G-2 -> G-2
    12: 8,    # G#-2 -> G#-2
    13: 9,    # A-2 -> A-2
    14: 10,   # Bb-2 -> Bb-2
    15: 11,   # B-2 -> B-2
    16:12,
    17:13,
    18:14,
    19:15,
    24:16,
    25:17,
    26:18,
    27:19,
    28:20,
    29:21,
    30:22,
    31:23,
    36:24,
    37:25,
    38:26,
    39:27,
    40:28,
    41:29,
    42:30,
    43:31,
    48:32,
    49:33,
    50:34,
    51:35,
    52:36,
    53:37,
    60:38,
    61:39,
    62:40,
    72:41,
    73:42,
    74:43,
    85:44,
    96:45,
    97:46,
    98:47,
    99:48,
    100:49,
    101:50,
    103:51,
    108:52,
    109:53,
}


def generic_to_chromatic(input_note):
    mapping = generic_to_chromatic_mapping
    return mapping.get(input_note, input_note)

def chromatic_to_generic(input_note):
    mapping = {v: k for k, v in generic_to_chromatic_mapping.items()}
    return mapping.get(input_note, input_note)

# Chromatic MIDI Pitch (0..53) -> Groove MIDI compressed pitch
# Built from generic_midi_mapping_compressed.csv sound names + best semantic match
CHROMATIC_TO_GROOVE_COMPRESSED = {
    0: 36,   # BASS open
    1: 36,   # BASS muted
    2: 36,   # BASS accent
    3: 36,   # BASS roll
    4: 48,   # High Tom
    5: 45,   # Mid Tom
    6: 43,   # Low Tom
    7: 43,   # Floor Tom
    8: 45,   # TONE low open
    9: 45,   # TONE low muted
    10: 48,  # TONE high open
    11: 48,  # TONE high muted
    12: 38,  # SLAP open/bright
    13: 40,  # RIM/edge
    14: 38,  # ROLL
    15: 38,  # FLAM/grace
    16: 36,  # DUM low
    17: 38,  # TAK high
    18: 40,  # SLAP sharp
    19: 40,  # MUTE/damped
    20: 40,  # RIM/edge
    21: 38,  # ROLL
    22: 42,  # JINGLE/shake
    23: 38,  # ORNAMENT/fill
    24: 38,  # HIT center
    25: 40,  # RIMSHOT
    26: 37,  # SIDE STICK
    27: 38,  # GHOST soft
    28: 38,  # BUZZ/press roll
    29: 38,  # FLAM
    30: 37,  # STICK CLICK
    31: 38,  # ACCENT hit
    32: 51,  # BELL low
    33: 51,  # BELL high
    34: 51,  # BELL low muted
    35: 51,  # BELL high muted
    36: 51,  # BELL roll
    37: 51,  # BELL accent
    38: 51,  # GONG large
    39: 51,  # GONG medium
    40: 51,  # GONG small
    41: 37,  # CLAVES
    42: 37,  # WOOBLOCK high
    43: 45,  # WOOBLOCK low
    44: 42,  # SHAKER accent
    45: 42,  # HI-HAT closed
    46: 46,  # HI-HAT open
    47: 44,  # HI-HAT pedal
    48: 51,  # RIDE
    49: 51,  # RIDE bell
    50: 49,  # CRASH
    51: 49,  # SPLASH/FX
    52: 38,  # HAND CLAP
    53: 38,  # DOUBLE CLAP
}


def chromatic_to_groove_compressed(input_note: int) -> int:
    return CHROMATIC_TO_GROOVE_COMPRESSED.get(input_note, input_note)


# Inverse: one groove class -> many chromatic candidates
GROOVE_COMPRESSED_TO_CHROMATIC = {}
for chrom_note, groove_note in CHROMATIC_TO_GROOVE_COMPRESSED.items():
    GROOVE_COMPRESSED_TO_CHROMATIC.setdefault(groove_note, []).append(chrom_note)


def groove_compressed_to_chromatic_candidates(groove_note: int) -> list[int]:
    return GROOVE_COMPRESSED_TO_CHROMATIC.get(groove_note, [groove_note])


def groove_compressed_to_chromatic(groove_note: int, strategy: str = "random") -> int:
    candidates = groove_compressed_to_chromatic_candidates(groove_note)
    if strategy == "first":
        return candidates[0]
    if strategy == "last":
        return candidates[-1]
    if strategy == "random":
        return random.choice(candidates)
    # default fallback
    return candidates[0]