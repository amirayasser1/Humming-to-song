# app.py
from __future__ import annotations
import streamlit as st
import tempfile
import os
import math
import librosa

from config import Config
from src.utils import list_files, safe_makedirs
from src.database import build_db, save_db, load_db, entry_to_rep
from src.midi_io import extract_melody_notes
from src.melody_repr import notes_to_rep
from src.similarity import dp_distance
from src.audio_query import wav_to_melody_rep, audio_window_to_rep

# -------------------------------
# Streamlit page config & style
# -------------------------------
st.set_page_config(
    page_title="Humming Song Finder",
    page_icon="🎵",
    layout="centered"
)

st.markdown("""
<style>
.stButton > button {
    background-color: #2196F3;
    color: white;
    font-weight: bold;
}
.result-item {
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
    background-color: #f5f5f5;
}
.best-match {
    background-color: #2e8b57;
    border-left: 5px solid #2196F3;
}


</style>
""", unsafe_allow_html=True)

# -------------------------------
# App title & uploader at the top
# -------------------------------
st.title("Humming Song Finder")
st.markdown("Upload your humming to find the matching song from your database.")

uploaded_file = st.file_uploader("Upload your humming (WAV/MP3)", type=['wav', 'mp3'])

# -------------------------------
# Sidebar settings (can stay on the side)
# -------------------------------
with st.sidebar:
    st.header("Settings")
    num_results = st.slider("Number of results to show", 1, 10, 5)
    win = st.number_input("Window length (s)", min_value=5.0, max_value=30.0, value=20.0, step=1.0)
    hop = st.number_input("Window hop (s)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
    max_sec = st.number_input("Max audio duration (s)", min_value=10.0, max_value=300.0, value=120.0, step=10.0)
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
    1. Prepare a database of songs (MIDI files) in `data/songs/`.
    2. Record yourself humming the melody.
    3. Upload the file.
    4. Click 'Find Matches' to see results.
    """)

cfg = Config()

# -------------------------------
# Auto-build or load database
# -------------------------------
db = None
if not os.path.exists(cfg.db_out):
    st.info("Database not found. Building automatically...")
    midi_paths = list_files(cfg.midi_dir, (".mid", ".midi"))
    if not midi_paths:
        st.error(f"No MIDI files found in {cfg.midi_dir} to build DB.")
    else:
        safe_makedirs(cfg.db_out)
        bad_log = "data/db/bad_midis.txt"
        with st.spinner("Building database..."):
            db = build_db(midi_paths, min_note_duration_s=cfg.min_note_duration_s, bad_log_path=bad_log)
            save_db(db, cfg.db_out)
        st.success(f"Database built! {len(db)} songs saved.")
else:
    db = load_db(cfg.db_out)
    st.info(f"Database loaded with {len(db)} songs.")

# -------------------------------
# Melody search function
# -------------------------------
def find_windowed_matches(audio_path, db, win, hop, max_sec, topk):
    duration = librosa.get_duration(path=audio_path)
    duration = min(duration, max_sec)

    starts = []
    s = 0.0
    while s + win <= duration:
        starts.append(s)
        s += hop
    if not starts:
        starts = [0.0]

    best_score = {entry.song_id: float("inf") for entry in db}
    best_cost  = {entry.song_id: float("inf") for entry in db}
    best_start = {entry.song_id: None for entry in db}
    best_len   = {entry.song_id: 0 for entry in db}
    usable = 0

    for start_s in starts:
        q_rep = audio_window_to_rep(audio_path, start_s=start_s, dur_s=win, sr=16000)
        L = len(q_rep.intervals)
        if L < 12:
            continue
        usable += 1
        for entry in db:
            sid = entry.song_id
            s_rep = entry_to_rep(entry)
            res = dp_distance(q_rep, s_rep)
            score = res.cost / math.sqrt(L)
            if score < best_score[sid]:
                best_score[sid] = score
                best_cost[sid] = res.cost
                best_start[sid] = start_s
                best_len[sid] = L

    ranked = []
    for sid in best_score:
        similarity = max(0.0, 1.0 - best_score[sid] / 5.0) * 100
        ranked.append((similarity, sid, best_cost[sid], best_len[sid], best_start[sid]))

    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[:topk], len(starts), usable

# -------------------------------
# Handle uploaded humming
# -------------------------------
if uploaded_file and db:
    col1, col2 = st.columns(2)
    with col1: st.metric("File", uploaded_file.name)
    with col2: st.metric("Size", f"{uploaded_file.size / 1024:.0f} KB")
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded_file.read())
        humming_path = tmp.name

    if st.button("🔍 Find Matching Songs"):
        with st.spinner("Processing audio..."):
            try:
                ranked, n_windows, usable = find_windowed_matches(humming_path, db, win, hop, max_sec, num_results)

                if usable == 0:
                    st.warning("No usable windows detected (need at least 12 intervals).")
                else:
                    st.info(f"Scanned {n_windows} windows, usable={usable}")

                    # Best match
                    best_sim, best_song, best_cost, L, start_s = ranked[0]
                    start_txt = f"{start_s:.1f}s" if start_s is not None else "n/a"
                    st.markdown(f"""
                    <div class='result-item best-match'>
                        <h3>🏆 Best Match: {best_song}</h3>
                        <h4>Similarity: {best_sim:.1f}%</h4>
                        <p>Length={L} intervals, start={start_txt}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Other matches
                    st.subheader("Other Matches:")
                    for i, (sim, song, cost, L, start_s) in enumerate(ranked, 1):
                        start_txt = f"{start_s:.1f}s" if start_s is not None else "n/a"
                        col_a, col_b = st.columns([4,1])
                        with col_a: st.write(f"**{i}. {song}** (L={L}, start={start_txt})")
                        with col_b: st.write(f"{sim:.1f}%")
                        st.progress(min(sim / 100.0, 0.99))

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                try: os.unlink(humming_path)
                except: pass
else:
    st.info("Upload a humming recording to get started.")
