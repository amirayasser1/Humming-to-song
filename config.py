from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    midi_dir: str = "data/midi_songs"
    db_out: str = "data/db/melody_db.json"

    # MIDI melody track selection heuristic
    # (you can improve later or manually override per-song)
    min_note_duration_s: float = 0.08  # remove tiny grace notes

    # Interval binning (paper-like 9 symbols)
    # We implement exact mapping in melody_repr.py
