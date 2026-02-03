from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple
from tqdm import tqdm

from .midi_io import extract_melody_notes
from .melody_repr import notes_to_rep, MelodyRep


@dataclass
class SongEntry:
    song_id: str
    midi_path: str
    pitches: List[int]
    intervals: List[int]
    contour: List[int]
    ioi: List[float]


def build_db(
    midi_paths: list[str],
    min_note_duration_s: float,
    bad_log_path: str = "data/db/bad_midis.txt",
) -> list[SongEntry]:
    """
    Build DB but skip corrupted / unreadable MIDI files.
    Writes skipped file paths + error messages to bad_log_path.
    """
    db: list[SongEntry] = []
    bad: list[Tuple[str, str]] = []

    for path in tqdm(midi_paths, desc="Building melody DB"):
        try:
            notes = extract_melody_notes(path, min_note_duration_s=min_note_duration_s)
            rep = notes_to_rep(notes)

            # Skip empty melodies (can happen if no valid melody track found)
            if len(rep.intervals) < 2:
                bad.append((path, "No usable melody extracted (too few notes/intervals)."))
                continue

            song_id = path.split("/")[-1].rsplit(".", 1)[0]
            db.append(SongEntry(
                song_id=song_id,
                midi_path=path,
                pitches=rep.pitches,
                intervals=rep.intervals,
                contour=rep.contour,
                ioi=rep.ioi,
            ))
        except Exception as e:
            bad.append((path, f"{type(e).__name__}: {e}"))
            continue

    # Write bad MIDI log
    if bad_log_path:
        import os
        os.makedirs(os.path.dirname(bad_log_path), exist_ok=True)
        with open(bad_log_path, "w", encoding="utf-8") as f:
            for p, msg in bad:
                f.write(f"{p}\t{msg}\n")

    return db


def save_db(db: list[SongEntry], out_path: str) -> None:
    payload = [entry.__dict__ for entry in db]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_db(path: str) -> list[SongEntry]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [SongEntry(**x) for x in payload]


def entry_to_rep(entry: SongEntry) -> MelodyRep:
    return MelodyRep(
        pitches=entry.pitches,
        intervals=entry.intervals,
        contour=entry.contour,
        ioi=entry.ioi
    )
