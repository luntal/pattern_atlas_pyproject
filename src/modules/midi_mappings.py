import random



def groove_midi_to_groove_midi_merged(input_note):
    mapping = {
        22: 42,  # HH Closed (Edge) -> Closed Hi-Hat (42)
        26: 46,  # HH Open (Edge) -> Open Hi-Hat (46)
        36: 36,  # Kick -> Kick (36)
        37: 37,  # Snare X-Stick -> Snare X-Stick(37)
        38: 38,  # Snare (Head) -> Snare (38)
        40: 40,  # Snare (Rim) -> Snare (40)
        42: 42,  # HH Closed (Bow) -> Closed Hi-Hat (42)
        43: 43,  # Tom 3 (Head) -> High Floor Tom (43)
        44: 44,  # HH Pedal -> HH Pedal (44)
        45: 45,  # Tom 2 -> Tom 2 (45)
        46: 46,  # HH Open (Bow) -> Open Hi-Hat (46)
        47: 45,  # Tom 2 (Rim) -> Tom 2 (45)
        48: 48,  # Tom 1 -> High Tom (48)
        49: 49,  # Crash 1 (Bow) -> Crash Cymbal (49)
        50: 48,  # Tom 1 (Rim) -> High Tom (48)
        51: 51,   # Ride (Bow) -> Ride Cymbal (51)
        52: 49,   # Crash 2 (Edge) -> Crash Cymbal (49)
        53: 51,   # Ride (Bell) -> Ride Cymbal (51)
        55: 49,   # Crash 1 (Edge) -> Crash Cymbal (49)
        57: 49,   # Crash 2 (Bow) -> Crash Cymbal (49)
        58: 43,   # Tom3(Rim) -> High Floor Tom(43)
        59: 51,   # Ride (Edge) -> Ride Cymbal (51)
    }
    return mapping.get(input_note, input_note)
'''
Merged mapping:
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
Ride (Bow) -> 51
'''



def groove_midi_merged_to_groove_midi_zeroed(input_note):
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
        51: 11,
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
CHROMATIC_TO_GROOVE_MERGED = {
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
    24: 38,  # SNARE HIT center
    25: 40,  # SNARE RIMSHOT
    26: 37,  # SNARE SIDE STICK
    27: 38,  # SNARE GHOST soft
    28: 38,  # SNARE BUZZ/press roll
    29: 38,  # SNARE FLAM
    30: 37,  # SNARE STICK CLICK
    31: 38,  # SNARE ACCENT hit
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



def chromatic_to_groove_merged(input_note: int) -> int:
    return CHROMATIC_TO_GROOVE_MERGED.get(input_note, input_note)


# Inverse: one groove class -> many chromatic candidates
GROOVE_MERGED_TO_CHROMATIC = {}
for chrom_note, groove_note in CHROMATIC_TO_GROOVE_MERGED.items():
    GROOVE_MERGED_TO_CHROMATIC.setdefault(groove_note, []).append(chrom_note)


def groove_merged_to_chromatic_candidates(groove_note: int) -> list[int]:
    return GROOVE_MERGED_TO_CHROMATIC.get(groove_note, [groove_note])


def groove_merged_to_chromatic(groove_note: int, strategy: str = "random") -> int:
    candidates = groove_merged_to_chromatic_candidates(groove_note)
    if strategy == "first":
        return candidates[0]
    if strategy == "last":
        return candidates[-1]
    if strategy == "random":
        return random.choice(candidates)
    # default fallback
    return candidates[0]


# ---------------------------------------------------------------------------
# Feature-based fan-out: merged Groove MIDI class → ordered chromatic bank
# ---------------------------------------------------------------------------
# Use ALL chromatic candidates for each merged class so every pitch in the
# compressed generic mapping can be reached.
_BASE_BANKS_FROM_INVERSE: dict[int, list[int]] = {
    merged_note: sorted(chromatic_notes)
    for merged_note, chromatic_notes in GROOVE_MERGED_TO_CHROMATIC.items()
}

# Preferred ordering for musically central articulations; remaining candidates
# are appended automatically to keep full coverage.
_BANK_PRIORITY: dict[int, list[int]] = {
    36: [2, 0, 1, 3],
    37: [26, 30, 41, 42],
    38: [31, 24, 12, 29, 27, 17, 14, 23, 52, 53],
    40: [25, 18, 13, 20, 19],
    42: [45, 22, 44],
    43: [7, 6],
    # Spread HH pedal-heavy events across low-percussive alternatives.
    44: [47, 43, 8, 9],
    45: [5, 8, 9, 43],
    46: [46],
    48: [4, 10, 11],
    49: [50, 51],
    51: [48, 49, 37, 33, 32, 36, 35, 34, 39, 38, 40],
}

_BANK_EXTRAS: dict[int, list[int]] = {
    # Requested manual expansion for HH pedal class.
    44: [43, 8, 9],
}

GROOVE_MERGED_FANOUT_BANKS: dict[int, list[int]] = {}
for merged_note, base_bank in _BASE_BANKS_FROM_INVERSE.items():
    priority = _BANK_PRIORITY.get(merged_note, [])
    ordered = [p for p in priority if p in base_bank]
    ordered.extend([p for p in base_bank if p not in ordered])
    for extra in _BANK_EXTRAS.get(merged_note, []):
        if extra not in ordered:
            ordered.append(extra)
    GROOVE_MERGED_FANOUT_BANKS[merged_note] = ordered


def _resolve_fanout_bank(merged_note: int, variant_index: int = 0) -> list[int]:
    """Return the variant-specific bank order for a merged class.

    We keep class semantics (same candidates) but rotate order per file variant
    to avoid always favoring the same chromatic lane for the loudest hit.
    """
    base_bank = GROOVE_MERGED_FANOUT_BANKS.get(merged_note, [merged_note])
    if len(base_bank) <= 1:
        return base_bank

    # Class-dependent salt keeps classes from rotating in lockstep.
    shift = (variant_index + (merged_note % len(base_bank))) % len(base_bank)
    rotated = base_bank[shift:] + base_bank[:shift]

    # Alternate direction for extra spread while remaining deterministic.
    if variant_index % 2 == 1 and len(base_bank) > 2:
        rotated = list(reversed(rotated))

    return rotated


def fan_out_merged_step(
    step_events: list[dict],
    variant_index: int = 0,
    state: dict | None = None,
    fixed_map: dict[int, int] | None = None,
) -> list[dict]:
    """Assign chromatic pitches (0-53) to merged-class events at a single time step.

    Events sharing the same merged class are sorted by velocity (descending) and
    assigned consecutive entries from that class's fan-out bank without replacement.
    The loudest hit receives the most prominent chromatic variant, preserving
    intra-class structure (e.g. bass accent vs. bass muted) even when multiple
    variants of the same merged class fire simultaneously.

    Args:
        step_events: list of dicts, each with at least:
            ``merged_note`` (int)  – groove merged pitch (36/37/38/40/42/43/44/45/46/48/49/51)
            ``velocity``    (int)  – raw MIDI velocity 0-127
            Any additional keys are passed through unchanged.
        variant_index: deterministic fan-out variant index used to rotate bank
            priority for a file-specific mapping strategy.
        state: optional mutable state dict used to improve long-range balance.
            Expected keys:
            - ``merged_cursors``: dict[int, int]
            - ``pitch_usage``: dict[int, int]
            If omitted, behavior is deterministic but stateless.
        fixed_map: optional mapping ``merged_note -> chromatic_note``. If a
            merged class exists in this map, that exact chromatic target is
            reused for all occurrences (useful to keep mapping stable within
            a 2-bar window).

    Returns:
        New list of dicts with ``chromatic_note`` (int, 0-53) added to each entry.
    """
    grouped: dict[int, list[int]] = {}
    for i, event in enumerate(step_events):
        grouped.setdefault(event['merged_note'], []).append(i)

    result = [dict(event) for event in step_events]
    used_in_step: set[int] = set()

    if state is None:
        state = {}
    merged_cursors: dict[int, int] = state.setdefault('merged_cursors', {})
    pitch_usage: dict[int, int] = state.setdefault('pitch_usage', {})

    for merged_note, indices in grouped.items():
        if fixed_map is not None and merged_note in fixed_map:
            fixed_pitch = fixed_map[merged_note]
            for idx in indices:
                result[idx]['chromatic_note'] = fixed_pitch
                used_in_step.add(fixed_pitch)
            continue

        bank = _resolve_fanout_bank(merged_note, variant_index=variant_index)
        # Highest velocity → most prominent (index 0) bank entry
        sorted_indices = sorted(indices, key=lambda i: step_events[i]['velocity'], reverse=True)

        cursor = merged_cursors.get(merged_note, 0)
        selected_for_merged = None
        for rank, idx in enumerate(sorted_indices):
            # Build deterministic candidate order from cursor + rank.
            base = (cursor + rank) % len(bank)
            ordered = [bank[(base + offset) % len(bank)] for offset in range(len(bank))]

            # Prefer globally underused lanes first, but do not duplicate in same step
            # unless we run out of unique candidates.
            unique_ordered = [candidate for candidate in ordered if candidate not in used_in_step]
            if not unique_ordered:
                unique_ordered = ordered

            chosen = min(unique_ordered, key=lambda candidate: (pitch_usage.get(candidate, 0), ordered.index(candidate)))
            result[idx]['chromatic_note'] = chosen
            used_in_step.add(chosen)
            pitch_usage[chosen] = pitch_usage.get(chosen, 0) + 1
            if selected_for_merged is None:
                selected_for_merged = chosen

        merged_cursors[merged_note] = (cursor + len(sorted_indices)) % len(bank)
        if fixed_map is not None and selected_for_merged is not None:
            fixed_map[merged_note] = selected_for_merged

    return result