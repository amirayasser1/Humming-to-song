from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa
from scipy.ndimage import median_filter

from .midi_io import NoteEvent
from .melody_repr import notes_to_rep, MelodyRep


@dataclass
class PitchTrack:
    times: np.ndarray      # shape (T,)
    f0: np.ndarray         # shape (T,) Hz, 0 for unvoiced


def load_audio_window(path: str, sr: int = 16000, start_s: float = 0.0, dur_s: float = 12.0) -> tuple[np.ndarray, int]:
    """
    Load a window of audio [start_s, start_s+dur_s].
    """
    y, sr = librosa.load(path, sr=sr, mono=True, offset=float(start_s), duration=float(dur_s))
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y, sr


def estimate_f0_yin(y: np.ndarray, sr: int,
                    fmin: float = 80.0, fmax: float = 500.0,
                    frame_length: int = 2048, hop_length: int = 256) -> PitchTrack:
    """
    DSP pitch tracking using YIN (no ML).
    """
    f0 = librosa.yin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        frame_length=frame_length, hop_length=hop_length
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

    # Simple voicing via RMS threshold
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    thr = np.percentile(rms, 35)
    voiced = rms > thr
    f0 = np.where(voiced, f0, 0.0)

    # Smooth spikes
    f0_s = median_filter(f0, size=7)
    return PitchTrack(times=times, f0=f0_s)


def hz_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def segment_notes_from_f0(track: PitchTrack,
                          min_note_dur_s: float = 0.12,
                          cents_change: float = 70.0) -> list[NoteEvent]:
    """
    Convert framewise f0 into note events.
    - Split by unvoiced frames
    - Split by pitch change > cents_change
    """
    times = track.times
    f0 = track.f0

    midi = np.full_like(f0, np.nan, dtype=float)
    voiced_idx = f0 > 0
    midi[voiced_idx] = hz_to_midi(f0[voiced_idx])

    notes: list[NoteEvent] = []
    i = 0
    T = len(times)

    while i < T:
        if np.isnan(midi[i]):
            i += 1
            continue

        start = i
        j = i + 1
        while j < T and not np.isnan(midi[j]):
            j += 1

        k = start
        while k < j:
            seg_start = k
            seg_ref = midi[k]
            k += 1

            while k < j:
                if abs(midi[k] - seg_ref) * 100.0 > cents_change:
                    break
                seg_ref = 0.92 * seg_ref + 0.08 * midi[k]
                k += 1

            seg_end = k
            onset_t = float(times[seg_start])
            end_t = float(times[seg_end - 1])
            dur = max(0.0, end_t - onset_t)

            if dur >= min_note_dur_s:
                seg_midi_med = float(np.nanmedian(midi[seg_start:seg_end]))
                pitch_int = int(np.round(seg_midi_med))
                notes.append(NoteEvent(pitch=pitch_int, onset=onset_t, duration=dur))

        i = j

    # Merge near-identical consecutive notes separated by tiny gaps
    merged: list[NoteEvent] = []
    for ev in notes:
        if not merged:
            merged.append(ev)
            continue
        prev = merged[-1]
        gap = ev.onset - (prev.onset + prev.duration)
        if abs(ev.pitch - prev.pitch) <= 1 and gap < 0.05:
            new_end = max(prev.onset + prev.duration, ev.onset + ev.duration)
            merged[-1] = NoteEvent(pitch=prev.pitch, onset=prev.onset, duration=new_end - prev.onset)
        else:
            merged.append(ev)

    return merged


def audio_window_to_rep(path: str, start_s: float, dur_s: float = 12.0, sr: int = 16000) -> MelodyRep:
    """
    Convert one audio window to MelodyRep (intervals + contour + IOIs).
    """
    y, sr = load_audio_window(path, sr=sr, start_s=start_s, dur_s=dur_s)
    track = estimate_f0_yin(y, sr=sr)
    notes = segment_notes_from_f0(track, min_note_dur_s=0.12, cents_change=70.0)
    rep = notes_to_rep(notes)
    return rep


def wav_to_melody_rep(path: str, sr: int = 16000) -> MelodyRep:
    """
    Single-shot (better for short clips). For long clips use windowed search.
    """
    y, sr = librosa.load(path, sr=sr, mono=True)
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    track = estimate_f0_yin(y, sr=sr)
    notes = segment_notes_from_f0(track, min_note_dur_s=0.12, cents_change=70.0)
    return notes_to_rep(notes)


def get_window_rep_debug(path: str, start_s: float, dur_s: float = 12.0) -> dict:
    rep = audio_window_to_rep(path, start_s=start_s, dur_s=dur_s, sr=16000)
    return {
        "start_s": start_s,
        "dur_s": dur_s,
        "n_intervals": len(rep.intervals),
        "intervals": rep.intervals[:80],
        "contour": rep.contour[:80],
        "ioi": [round(x, 3) for x in rep.ioi[:80]],
    }
