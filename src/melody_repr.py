from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .midi_io import NoteEvent


@dataclass
class MelodyRep:
    pitches: List[int]     # per-note MIDI pitches (length = #notes)
    intervals: List[int]   # coarse symbols in [-4..+4] (length = #notes-1)
    contour: List[int]     # -1 down, 0 same, +1 up (length = #notes-1)
    ioi: List[float]       # inter-onset intervals seconds (length = #notes-1)


def bin_interval_semitones(semi: int) -> int:
    """
    Paper-like 9-bin mapping.
    Returns symbol in {-4,-3,-2,-1,0,+1,+2,+3,+4}.
    """
    if semi <= -7:
        return -4
    if semi in (-6, -5):
        return -3
    if semi in (-4, -3):
        return -2
    if semi in (-2, -1):
        return -1
    if semi == 0:
        return 0
    if semi in (1, 2):
        return 1
    if semi in (3, 4):
        return 2
    if semi in (5, 6):
        return 3
    if semi >= 7:
        return 4
    return 0


def _contour_from_semitones(semi: int) -> int:
    if semi > 0:
        return 1
    if semi < 0:
        return -1
    return 0


def notes_to_rep(notes: list[NoteEvent]) -> MelodyRep:
    if len(notes) < 2:
        pitches = [n.pitch for n in notes]
        return MelodyRep(pitches=pitches, intervals=[], contour=[], ioi=[])

    pitches = [n.pitch for n in notes]
    onsets = [n.onset for n in notes]

    raw_intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
    intervals = [bin_interval_semitones(int(x)) for x in raw_intervals]
    contour = [_contour_from_semitones(int(x)) for x in raw_intervals]

    ioi = [max(1e-6, onsets[i + 1] - onsets[i]) for i in range(len(onsets) - 1)]

    return MelodyRep(pitches=pitches, intervals=intervals, contour=contour, ioi=ioi)
