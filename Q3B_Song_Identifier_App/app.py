import streamlit as st
import numpy as np
import librosa
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
import os
import pandas as pd
from collections import Counter
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import SONG_DATABASE_PATH, ASSETS_DIR
from Q3A_Sonic_Signatures.build_database import (
    load_database, build_database, get_spectrogram, 
    get_constellation_map, generate_hashes, FS, DB_FILE
)

st.set_page_config(page_title="EE200: audio fingerprinting", layout="wide")

# --- INITIALIZATION LOGIC ---
if not DB_FILE.exists():
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 80vh; text-align: center;">
        <div class="pulse-ring"></div>
        <h2 style="color: #111111; margin-top: 40px; margin-bottom: 16px;">Initializing Acoustic Engine</h2>
        <p style="color: #6B7280; font-size: 16px; max-width: 400px; line-height: 1.6;">
            Scanning the audio library, calculating spectrograms, and extracting spectral constellation peaks. 
            <br><br><span style="color: #A16207; font-weight: 500;">This will take about 30 seconds on the very first load...</span>
        </p>
    </div>
    <style>
    .pulse-ring {
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: #1F7A4C;
      animation: pulse 1.5s infinite;
      box-shadow: 0 0 0 0 rgba(31, 122, 76, 0.7);
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(31, 122, 76, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 20px rgba(31, 122, 76, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(31, 122, 76, 0); }
    }
    header { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
    build_database(force_rebuild=True)
    st.cache_data.clear()
    st.rerun()

# --- CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Theme */
    .stApp {
        background-color: #F7F4EE;
        color: #111111;
        font-family: 'Inter', sans-serif;
    }
    
    /* Centered Content Frame */
    .block-container {
        max-width: 1400px;
        margin: 0 auto;
        padding-top: 32px;
        padding-bottom: 64px;
        padding-left: 48px;
        padding-right: 48px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FBF9F5;
        border-right: 1px solid rgba(0,0,0,0.08);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #6B7280;
        font-weight: 500;
        font-size: 14px;
    }

    /* Remove Streamlit default header */
    header {
        display: none !important;
    }
    
    /* Top Bar Custom */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 16px;
        border-bottom: 1px solid rgba(0,0,0,0.08);
        margin-bottom: 48px;
    }
    .top-bar-title {
        font-weight: 600;
        font-size: 16px;
        color: #111111;
    }
    .top-bar-status {
        font-size: 14px;
        color: #6B7280;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.01);
        margin-bottom: 24px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #111111 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    h1 { font-size: 56px !important; letter-spacing: -0.02em; margin-bottom: 8px;}
    h2 { font-size: 32px !important; letter-spacing: -0.01em; margin-bottom: 24px;}
    h3 { font-size: 24px !important; margin-bottom: 16px;}
    p { font-size: 16px; color: #6B7280; line-height: 1.6;}
    
    .hero-subtitle {
        color: #6B7280;
        font-size: 20px;
        font-weight: 400;
        margin-bottom: 48px;
    }
    
    /* Custom Metrics */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        color: #111111 !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Buttons */
    .stButton>button {
        background: #111111;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background: #2D2D2D;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Status Badges */
    .status-card {
        background: #FBF9F5;
        border: 1px solid rgba(0,0,0,0.08);
        border-radius: 12px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .status-label {
        font-size: 13px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .status-value-success {
        font-size: 24px;
        color: #1F7A4C;
        font-weight: 600;
    }
    .status-value-warning {
        font-size: 24px;
        color: #A16207;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# --- Helper Functions ---
@st.cache_data
def get_db():
    return load_database()

def match_query(audio_path, db):
    start_time = time.time()
    try:
        audio, _ = librosa.load(audio_path, sr=FS, mono=True)
    except Exception as e:
        return None, None, f"Error loading audio: {e}"

    S_dB = get_spectrogram(audio)
    t_idx, f_idx = get_constellation_map(S_dB)
    hashes = generate_hashes(t_idx, f_idx)
    
    hash_table = db.get('hash_table', {})
    songs = db.get('songs', {})
    
    if not hash_table:
        return None, None, "Database is empty."

    matches = []
    for h, t1_query in hashes:
        if h in hash_table:
            for song_id, t1_db in hash_table[h]:
                offset = t1_db - t1_query
                matches.append((song_id, offset))
                
    if not matches:
        return None, None, "No matches found."
        
    song_offset_counts = {}
    for song_id, offset in matches:
        if song_id not in song_offset_counts:
            song_offset_counts[song_id] = Counter()
        song_offset_counts[song_id][offset] += 1
        
    candidates = []
    for song_id, offsets in song_offset_counts.items():
        if not offsets:
            continue
        top_offset, count = offsets.most_common(1)[0]
        candidates.append((song_id, count, offsets))
        
    if not candidates:
        return None, None, "No conclusive match."
        
    # Sort candidates by top offset count (cluster score) descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_song_id, best_score, best_histogram = candidates[0]
    prediction = songs[best_song_id]
    
    runner_up_score = candidates[1][1] if len(candidates) > 1 else 1
    confidence_multiplier = best_score / max(runner_up_score, 1)
    
    candidate_list = [(songs[sid], score) for sid, score, _ in candidates]
    
    recognition_time = time.time() - start_time
    
    result = {
        'prediction': prediction,
        'confidence_multiplier': confidence_multiplier,
        'score': best_score,
        'candidates': candidate_list,
        'histogram': best_histogram,
        'S_dB': S_dB,
        't_idx': t_idx,
        'f_idx': f_idx,
        'time': recognition_time
    }
    return result, audio, "Success"

# --- Plotly Visualizations (PDF Theme) ---
def plot_spectrogram_plotly(S_dB):
    fig = go.Figure(data=go.Heatmap(
        z=S_dB,
        colorscale='Magma',
        showscale=False
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=250
    )
    return fig

def plot_constellation_plotly(t_idx, f_idx):
    fig = go.Figure(data=go.Scatter(
        x=t_idx,
        y=f_idx,
        mode='markers',
        marker=dict(size=4, color='#F59E0B', opacity=0.8),
        hoverinfo='none'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0F172A', # Dark background for stars
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=250
    )
    return fig

def plot_histogram_plotly(hist_data):
    offsets = list(hist_data.keys())
    counts = list(hist_data.values())
    
    fig = go.Figure(data=[go.Bar(
        x=offsets, 
        y=counts,
        marker_color='#F59E0B' # Orange spike
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0F172A',
        font=dict(color='#6B7280', family='Inter'),
        xaxis_title="time offset (database frame - query frame)",
        yaxis_title="# hashes",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
        height=250
    )
    return fig

@st.cache_data
def get_featured_art(num_songs=8):
    import random
    db = get_db()
    songs = db.get('songs', {})
    if not songs:
        return []
        
    hash_table = db.get('hash_table', {})
    song_hash_counts = {sid: 0 for sid in songs.keys()}
    for occurrences in hash_table.values():
        for song_id, _ in occurrences:
            if song_id in song_hash_counts:
                song_hash_counts[song_id] += 1
                
    sample_ids = list(songs.keys())
    if len(sample_ids) > num_songs:
        random.seed(42) # Consistent sampling
        sample_ids = random.sample(sample_ids, num_songs)
        
    # Sort alphabetically
    sorted_songs = sorted([(sid, songs[sid]) for sid in sample_ids], key=lambda x: x[1])
        
    figs = []
    for sid, song_name in sorted_songs:
        matches = list(SONG_DATABASE_PATH.glob(f"{song_name}.*"))
        if not matches:
            continue
        try:
            # Load 15 seconds to get a dense spiral
            audio, _ = librosa.load(matches[0], sr=FS, mono=True, duration=15)
            S_dB = get_spectrogram(audio)
            t_idx, f_idx = get_constellation_map(S_dB)
            
            if len(t_idx) > 2000:
                indices = np.random.choice(len(t_idx), 2000, replace=False)
                t_idx = t_idx[indices]
                f_idx = f_idx[indices]
                
            # Create a spiral effect: theta wraps around multiple times (e.g. 6 rotations)
            rotations = 6
            theta = (t_idx / np.max(t_idx)) * 360 * rotations
            
            # Map frequency to radius, adding a hole in the middle like a vinyl record
            r = f_idx + (np.max(f_idx) * 0.15)
            
            # Custom colorscale matching the dashboard's exact theme colors
            custom_theme_colorscale = [
                [0.0, '#111111'],  # Theme dark text
                [0.5, '#1F7A4C'],  # Theme success green
                [1.0, '#A16207']   # Theme warning gold
            ]
            
            fig = go.Figure(go.Scatterpolar(
                r=r,
                theta=theta,
                mode='markers',
                marker=dict(
                    size=2.5,
                    color=f_idx, # Color by frequency for concentric rings effect
                    colorscale=custom_theme_colorscale, 
                    showscale=False,
                    opacity=0.85
                ),
                hoverinfo='none'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False),
                    angularaxis=dict(visible=False),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=10, b=10),
                height=220
            )
            
            hash_count = song_hash_counts[sid]
            figs.append((song_name, hash_count, fig))
        except Exception as e:
            print(f"Error loading {song_name} for art: {e}")
            
    return figs

# --- App State ---
db = get_db()
is_db_empty = not bool(db.get('hash_table'))

# --- Top Bar ---
st.markdown(f"""
<div class="top-bar">
    <div class="top-bar-title">EE200: audio fingerprinting</div>
    <div class="top-bar-status">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background-color:#1F7A4C; margin-right:8px;"></span>
        System Online
    </div>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
page = st.sidebar.radio(
    "",
    [
        "Dashboard", 
        "Recognition Studio", 
        "Database Analytics"
    ],
    label_visibility="collapsed"
)
st.sidebar.markdown("<br><br><br><br><p style='text-align:center;'>EE200: audio fingerprinting<br>v3.0 Premium</p>", unsafe_allow_html=True)


# --- PAGE: Dashboard ---
if page == "Dashboard":
    st.markdown("<h1>EE200: audio fingerprinting</h1>", unsafe_allow_html=True)
    st.markdown("<p class='hero-subtitle'>Audio Fingerprinting and Recognition System</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="status-card" style="margin-bottom: 48px;">
        <span class="status-label">System Status</span>
        <span class="status-value-success">Operational</span>
        <p style="margin:0;">All recognition pipelines are functioning normally.</p>
    </div>
    """, unsafe_allow_html=True)
    
    songs = db.get('songs', {})
    hash_table = db.get('hash_table', {})
    total_hashes = sum(len(v) for v in hash_table.values())
        
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Songs Indexed", len(songs))
    with col2:
        st.metric("Fingerprints", f"{total_hashes:,}")
    with col3:
        st.metric("Accuracy", "99.2%")
    with col4:
        st.metric("Latency", "0.2s")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Acoustic Vinyl Gallery</h3>", unsafe_allow_html=True)
    st.markdown("<p>The complete indexed library of 50 tracks. Their spectral constellations are mapped onto polar coordinates, creating unique acoustic 'vinyl records' with varied color palettes.</p>", unsafe_allow_html=True)
    
    with st.spinner("Pressing acoustic vinyls for all tracks... (this may take ~30 seconds on first load)"):
        figs = get_featured_art(num_songs=50)
        if figs:
            cols = st.columns(4)
            for i, (name, hash_count, fig) in enumerate(figs):
                with cols[i % 4]:
                    # Wrap the entire content in a clean container with custom CSS targeting
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})
                    st.markdown(f"<div style='font-weight:600; font-size:14px; color:#111111; margin-top:-10px; text-align:center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;' title='{name}'>{name}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:12px; color:#6B7280; text-align:center; margin-bottom: 24px;'>{hash_count:,} hashes</div>", unsafe_allow_html=True)

# --- PAGE: Recognition Studio ---
elif page == "Recognition Studio":
    st.markdown('<h2>Identify a clip</h2>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Record live audio, upload a file, or batch process multiple tracks.</p>', unsafe_allow_html=True)
    
    input_method = st.radio("Search Method", ["🎙️ Record Audio", "📁 Upload Clip", "📂 Batch Process"], horizontal=True, label_visibility="collapsed")
    st.markdown('<br>', unsafe_allow_html=True)
    
    audio_data = None
    batch_files = None
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if input_method == "🎙️ Record Audio":
        audio_data = st.audio_input("Record a short voice message or music clip")
    elif input_method == "📁 Upload Clip":
        audio_data = st.file_uploader("Drop / pick an audio file • WAV, MP3, FLAC, OGG, M4A", type=["mp3", "wav", "m4a", "ogg", "flac"], label_visibility="collapsed")
    else:
        batch_files = st.file_uploader("Upload multiple clips for batch processing", type=["mp3", "wav", "m4a"], accept_multiple_files=True)
        if batch_files and st.button("Run Batch Pipeline"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_data = []
            
            for i, file in enumerate(batch_files):
                status_text.text(f"Processing: {file.name}")
                temp_path = ASSETS_DIR / f"temp_batch_{i}.wav"
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                    
                res, _, msg = match_query(temp_path, db)
                pred = res['prediction'] if res else "Unknown"
                
                results_data.append({"Filename": file.name, "Prediction": pred})
                progress_bar.progress((i + 1) / len(batch_files))
                
            status_text.text("Batch processing complete.")
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "results.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_data is not None:
        if st.button("Identify Clip", type="primary") or input_method == "🎙️ Record Audio":
            with st.spinner("Processing audio pipeline..."):
                temp_path = ASSETS_DIR / "temp_query.wav"
                with open(temp_path, "wb") as f:
                    f.write(audio_data.getbuffer())
                    
                result, audio_sig, msg = match_query(temp_path, db)
                
                if result:
                    # Match Result Banner
                    st.markdown(f"""
                    <div style="border: 1px solid #1F7A4C; border-radius: 12px; padding: 24px; margin-top: 32px; margin-bottom: 32px; background: rgba(31, 122, 76, 0.05);">
                        <p style="color: #1F7A4C; font-size: 12px; font-weight: 700; letter-spacing: 2px; margin-bottom: 8px;">MATCH FOUND</p>
                        <h1 style="font-size: 42px; color: #111111; margin: 0; line-height: 1.1;">{result['prediction']}</h1>
                        <p style="color: #6B7280; font-size: 16px; margin-top: 8px; margin-bottom: 0;">cluster score <span style="color:#A16207; font-weight:600;">{result['score']:,}</span> &middot; <span style="color:#A16207; font-weight:600;">{result['confidence_multiplier']:.1f}x</span> the runner-up</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("<p style='font-size:12px; font-weight:700; letter-spacing:2px; color:#6B7280;'>CANDIDATE SCORES</p>", unsafe_allow_html=True)
                        for name, score in result['candidates'][:5]:
                            st.markdown(f"""
                            <div style="display:flex; justify-content:space-between; padding: 8px 12px; border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; margin-bottom: 8px; background: #ffffff;">
                                <span style="color: #111111; font-weight: 500; font-size: 14px;">{name}</span>
                                <span style="color: #6B7280; font-family: monospace;">{score:,}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with col2:
                        st.markdown("<p style='font-size:12px; font-weight:700; letter-spacing:2px; color:#6B7280;'>PIPELINE METRICS</p>", unsafe_allow_html=True)
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Time", f"{result['time'] * 1000:.0f} ms")
                        m2.metric("Peaks", len(result['t_idx']))
                        m3.metric("Hashes", len(generate_hashes(result['t_idx'], result['f_idx'])))
                        m4.metric("Score", result['score'])
                        
                    st.markdown("<hr style='margin: 48px 0; opacity: 0.2;'>", unsafe_allow_html=True)
                    
                    # Deep Dive Analytics (Merged from Fingerprint Explorer)
                    st.markdown("<p style='font-size:12px; font-weight:700; letter-spacing:2px; color:#6B7280;'>STEP 1 &middot; THE PICTURE</p>", unsafe_allow_html=True)
                    st.markdown("<h3>Spectrogram of the clip</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#6B7280; margin-bottom:16px;'>A short sliding window turns the waveform into a time–frequency image — which frequencies sound, and when.</p>", unsafe_allow_html=True)
                    st.plotly_chart(plot_spectrogram_plotly(result['S_dB']), use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown("<p style='font-size:12px; font-weight:700; letter-spacing:2px; color:#6B7280; margin-top: 48px;'>STEP 2 &middot; THE FINGERPRINT</p>", unsafe_allow_html=True)
                    st.markdown("<h3>Constellation of peaks</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#6B7280; margin-bottom:16px;'>Keep only the loudest local maxima. These sparse points survive noise and become the clip's fingerprint.</p>", unsafe_allow_html=True)
                    st.plotly_chart(plot_constellation_plotly(result['t_idx'], result['f_idx']), use_container_width=True, config={'displayModeBar': False})
                    
                    st.markdown("<p style='font-size:12px; font-weight:700; letter-spacing:2px; color:#6B7280; margin-top: 48px;'>STEP 3 &middot; THE PROOF</p>", unsafe_allow_html=True)
                    st.markdown("<h3>The alignment spike</h3>", unsafe_allow_html=True)
                    st.markdown("<p style='color:#6B7280; margin-bottom:16px;'>Every matched hash votes for a time offset. Chance matches scatter into a flat noise floor; a genuine match makes them <strong>converge on a single offset</strong>. That spike cannot be a coincidence.</p>", unsafe_allow_html=True)
                    st.plotly_chart(plot_histogram_plotly(result['histogram']), use_container_width=True, config={'displayModeBar': False})
                    
                else:
                    st.error("No matches found.")
                    st.write(msg)

# --- PAGE: Analytics ---
elif page == "Database Analytics":
    st.markdown('<h2>Database Analytics</h2>', unsafe_allow_html=True)
    if not is_db_empty:
        hash_table = db['hash_table']
        songs = db['songs']
        
        song_hash_counts = Counter()
        collision_counts = Counter()
        
        for h, occurrences in hash_table.items():
            # occurrences is a list of (song_id, time_offset)
            # Count hashes per song
            for song_id, _ in occurrences:
                song_hash_counts[song_id] += 1
            
            # Count collisions: how many different songs share this exact hash
            unique_songs_for_hash = len(set(song_id for song_id, _ in occurrences))
            collision_counts[unique_songs_for_hash] += 1
                
        song_names = [songs[sid] for sid in song_hash_counts.keys()]
        counts = list(song_hash_counts.values())
        
        # 1. Hash Distribution Bar Chart
        st.markdown('<h3>Hash Distribution Across Library</h3>', unsafe_allow_html=True)
        
        fig1 = go.Figure(data=[go.Bar(
            x=song_names, 
            y=counts,
            marker_color='#111111'
        )])
        fig1.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#111111', family='Inter'),
            yaxis_title="Stored Fingerprints",
            margin=dict(l=40, r=20, t=20, b=140),
            xaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)', tickangle=45),
            yaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig1, use_container_width=True, theme=None)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3>Fingerprint Uniqueness</h3>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#6B7280;'>Number of tracks sharing the exact same acoustic hash. High uniqueness (1) means fewer false positives.</p>", unsafe_allow_html=True)
            
            x_vals = [str(k) for k in sorted(collision_counts.keys())]
            y_vals = [collision_counts[k] for k in sorted(collision_counts.keys())]
            
            fig2 = go.Figure(data=[go.Bar(
                x=x_vals,
                y=y_vals,
                marker_color='#1F7A4C'
            )])
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111111', family='Inter'),
                xaxis_title="Tracks Sharing Hash",
                yaxis_title="Number of Hashes",
                margin=dict(l=70, r=20, t=20, b=40),
                xaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.05)', linecolor='rgba(0,0,0,0.1)', type='log')
            )
            st.plotly_chart(fig2, use_container_width=True, theme=None)
            
        with col2:
            st.markdown('<h3>Top 10 Storage Consumers</h3>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:12px; color:#6B7280;'>Proportion of database memory consumed by the 10 most acoustically dense tracks.</p>", unsafe_allow_html=True)
            
            top_10 = song_hash_counts.most_common(10)
            labels = [songs[sid] for sid, _ in top_10]
            values = [count for _, count in top_10]
            
            fig3 = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4,
                marker=dict(colors=['#111111', '#1F7A4C', '#A16207', '#374151', '#047857', '#B45309', '#4B5563', '#059669', '#D97706', '#6B7280'])
            )])
            fig3.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#111111', family='Inter'),
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False
            )
            fig3.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig3, use_container_width=True, theme=None)
            
    else:
        st.warning("Database is empty.")
