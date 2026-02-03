from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import os
import tempfile
import mido
import pretty_midi


@dataclass
class NoteEvent:
    pitch: int          # MIDI note number
    onset: float        # seconds
    duration: float     # seconds

def _remove_too_short(notes: list[pretty_midi.Note], min_dur: float) -> list[pretty_midi.Note]:
    out = []
    for n in notes:
        dur = max(0.0, n.end - n.start)
        if dur >= min_dur:
            out.append(n)
    return out

def _is_monophonic(notes: list[pretty_midi.Note]) -> bool:
    """Return True if no overlaps."""
    if not notes:
        return True
    notes_sorted = sorted(notes, key=lambda n: (n.start, n.end))
    last_end = notes_sorted[0].end
    for n in notes_sorted[1:]:
        if n.start < last_end - 1e-6:
            return False
        last_end = max(last_end, n.end)
    return True

def choose_melody_instrument(pm: pretty_midi.PrettyMIDI, min_note_duration_s: float = 0.08) -> Optional[pretty_midi.Instrument]:
    """
    Heuristic melody track selection:
      - prefer instruments that are mostly monophonic
      - prefer higher average pitch (melody tends to be higher than bass/chords)
      - prefer larger pitch variance (melody moves)
    """
    candidates = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes = _remove_too_short(inst.notes, min_note_duration_s)
        if len(notes) < 4:
            continue

        mono = _is_monophonic(notes)
        pitches = [n.pitch for n in notes]
        avg_pitch = sum(pitches) / len(pitches)
        var_pitch = sum((p - avg_pitch) ** 2 for p in pitches) / len(pitches)

        score = 0.0
        score += 3.0 if mono else 0.0
        score += avg_pitch / 30.0
        score += (var_pitch ** 0.5) / 10.0

        candidates.append((score, inst, notes))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def load_midi_robust(path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Attempt to load MIDI. If it fails, try to clean it using mido (clip velocities/data).
    """
    try:
        return pretty_midi.PrettyMIDI(path)
    except Exception as first_e:
        # Try cleaning with Mido
        try:
            mid = mido.MidiFile(path, clip=True)
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix=".mid")
            os.close(fd)
            mid.save(temp_path)
            
            try:
                pm = pretty_midi.PrettyMIDI(temp_path)
                return pm
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception:
            # If cleaning fails, return None (or raise original error)
            pass
        return None

def extract_melody_notes(midi_path: str, min_note_duration_s: float = 0.08) -> list[NoteEvent]:
    pm = load_midi_robust(midi_path)
    if pm is None:
        raise ValueError("Could not lead MIDI file (even after cleaning).")

    inst = choose_melody_instrument(pm, min_note_duration_s=min_note_duration_s)
    if inst is None:
        return []

    notes = _remove_too_short(inst.notes, min_note_duration_s)
    # SORT ORDER CHANGED: Sort by start time, then DESCENDING pitch
    # This ensures that if multiple notes start at the same time (chords),
    # the highest pitch comes first in the list.
    notes = sorted(notes, key=lambda n: (n.start, n.end, -n.pitch))

    melody: list[NoteEvent] = []
    for n in notes:
        dur = max(0.0, n.end - n.start)
        melody.append(NoteEvent(pitch=int(n.pitch), onset=float(n.start), duration=float(dur)))

    # Optional: make strictly monophonic by removing overlaps (keep earliest end)
    cleaned: list[NoteEvent] = []
    last_end = -1.0
    for ev in melody:
        if ev.onset >= last_end - 1e-6:
            cleaned.append(ev)
            last_end = ev.onset + ev.duration
        else:
            # overlap: skip (simple policy).
            # Because we sorted by -pitch, this keeps the highest note of a chord
            # and discards lower notes that start effectively at the same time.
            continue

    return cleaned
