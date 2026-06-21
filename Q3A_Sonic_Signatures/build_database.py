import os
import glob
import pickle
import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import SONG_DATABASE_PATH, ASSETS_DIR

# DB path
DB_FILE = ASSETS_DIR / "song_database.pkl"

# Fingerprinting parameters
FS = 22050
N_FFT = 2048
HOP_LENGTH = 512
PEAK_NEIGHBORHOOD_SIZE = 20 # for 2D max filter
DEFAULT_THRESHOLD = 10 # dB above background
TARGET_ZONE_T_MIN = 1
TARGET_ZONE_T_MAX = 50
TARGET_ZONE_F_MAX = 50

def get_spectrogram(audio, fs=FS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """Compute log-magnitude spectrogram"""
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    # Convert to dB
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    return S_dB

def get_constellation_map(S_dB, neighborhood_size=PEAK_NEIGHBORHOOD_SIZE, threshold=DEFAULT_THRESHOLD):
    """Find local maxima in spectrogram to form the constellation map"""
    # Find local maxima using a maximum filter
    local_max = maximum_filter(S_dB, size=neighborhood_size) == S_dB
    
    # Apply a noise threshold (only keep peaks that are prominent enough)
    background = np.median(S_dB)
    peaks = local_max & (S_dB > (background + threshold))
    
    # Get coordinates of peaks (freq_idx, time_idx)
    freq_idx, time_idx = np.where(peaks)
    
    # Sort by time
    sort_idx = np.argsort(time_idx)
    freq_idx = freq_idx[sort_idx]
    time_idx = time_idx[sort_idx]
    
    return time_idx, freq_idx

def generate_hashes(time_idx, freq_idx):
    """Generate pair-wise hashes from constellation map"""
    hashes = []
    num_peaks = len(time_idx)
    
    for i in range(num_peaks):
        for j in range(1, 15): # Look at next 15 peaks
            if (i + j) < num_peaks:
                t1 = time_idx[i]
                t2 = time_idx[i+j]
                f1 = freq_idx[i]
                f2 = freq_idx[i+j]
                
                t_delta = t2 - t1
                
                # Filter pairs by target zone
                if TARGET_ZONE_T_MIN <= t_delta <= TARGET_ZONE_T_MAX:
                    if abs(f2 - f1) <= TARGET_ZONE_F_MAX:
                        # hash tuple: (freq1, freq2, time_delta)
                        h = (f1, f2, t_delta)
                        # We store the absolute time t1 to compute offsets later
                        hashes.append((h, t1))
    return hashes

def build_database(force_rebuild=False):
    if DB_FILE.exists() and not force_rebuild:
        print(f"Database already exists at {DB_FILE}. Skipping rebuild.")
        return load_database()
        
    print(f"Scanning for audio files in {SONG_DATABASE_PATH}...")
    
    if not SONG_DATABASE_PATH.exists():
        print(f"WARNING: Database path {SONG_DATABASE_PATH} does not exist.")
        print("Please download the official song database to this location.")
        return {'hash_table': {}, 'songs': {}}

    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.m4a']:
        audio_files.extend(glob.glob(str(SONG_DATABASE_PATH / ext)))
        
    if not audio_files:
        print("No audio files found. Please ensure they are placed correctly.")
        return {'hash_table': {}, 'songs': {}}

    hash_table = {} # hash -> list of (song_id, t1)
    songs = {}      # song_id -> song_name

    for idx, fpath in enumerate(audio_files):
        song_name = Path(fpath).stem
        songs[idx] = song_name
        print(f"Processing ({idx+1}/{len(audio_files)}): {song_name}...")
        
        try:
            audio, _ = librosa.load(fpath, sr=FS, mono=True)
            S_dB = get_spectrogram(audio)
            t_idx, f_idx = get_constellation_map(S_dB)
            hashes = generate_hashes(t_idx, f_idx)
            
            for h, t1 in hashes:
                if h not in hash_table:
                    hash_table[h] = []
                hash_table[h].append((idx, t1))
        except Exception as e:
            print(f"Error processing {song_name}: {e}")

    db = {
        'hash_table': hash_table,
        'songs': songs
    }
    
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
        
    print(f"Database built! Indexed {len(songs)} songs with {len(hash_table)} unique hashes.")
    return db

def load_database():
    if not DB_FILE.exists():
        return build_database()
    with open(DB_FILE, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_database(force_rebuild=True)
