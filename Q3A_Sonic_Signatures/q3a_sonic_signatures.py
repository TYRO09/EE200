# %% [markdown]
# # Q3A: Sonic Signatures - 'Magical Mystery Tune'
# This notebook explores audio fingerprinting using a Shazam-like algorithm. We compare window lengths, implement constellation mapping, and analyze robustness against noise and pitch shifting.

# %% [markdown]
# ## 1. Setup and Import Libraries

# %%
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sys
from pathlib import Path

# Add project root to sys.path to import config
PROJECT_ROOT = Path('.').resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import SONG_DATABASE_PATH, ASSETS_DIR
from Q3A_Sonic_Signatures.build_database import get_spectrogram, get_constellation_map, FS

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)

OUT_DIR = ASSETS_DIR / "q3a"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## 2. Load Audio and Experiment with Window Lengths
# We compute spectrograms with different window lengths (n_fft) to observe the time-frequency resolution trade-off.
# *(Note: If the song database is missing, this cell will skip execution gracefully)*

# %%
audio_files = list(SONG_DATABASE_PATH.glob('*.mp3')) + list(SONG_DATABASE_PATH.glob('*.wav'))

if audio_files:
    test_song = audio_files[0]
    # Load first 10 seconds for demonstration
    y, sr = librosa.load(test_song, sr=FS, duration=10)
    
    # Short Window (e.g., n_fft = 512) -> Good time resolution, poor frequency resolution
    S_short = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=128)), ref=np.max)
    
    # Long Window (e.g., n_fft = 4096) -> Poor time resolution, good frequency resolution
    S_long = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)), ref=np.max)
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    librosa.display.specshow(S_short, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.title('Short Window (n_fft=512)\nSharp Time, Blurry Freq')
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(S_long, sr=sr, x_axis='time', y_axis='linear', cmap='magma')
    plt.title('Long Window (n_fft=4096)\nBlurry Time, Sharp Freq')
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "window_comparison.png", dpi=300)
    plt.show()
else:
    print("Song database not found. Please download it to the target directory to run experiments.")

# %% [markdown]
# **Observation on Time vs. Frequency Resolution:**
# A shorter window length creates a narrower "slice" in time, capturing rapid transient events like drum hits with excellent temporal accuracy, but blurring the specific pitch (frequency). Conversely, a long window can precisely resolve closely spaced pitches, but spreads short transient events across a wider time frame. 

# %% [markdown]
# ## 3. Constellation Map Visualization
# We extract the strongest peaks from the spectrogram to form the 'constellation' fingerprint.

# %%
if audio_files:
    # Standard parameters
    n_fft = 2048
    hop_length = 512
    S_dB = get_spectrogram(y, fs=FS, n_fft=n_fft, hop_length=hop_length)
    
    t_idx, f_idx = get_constellation_map(S_dB, neighborhood_size=20, threshold=15)
    
    # Convert indices to actual physical time/freq for plotting
    times = librosa.frames_to_time(t_idx, sr=FS, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=FS, n_fft=n_fft)[f_idx]
    
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_dB, sr=FS, hop_length=hop_length, x_axis='time', y_axis='linear', cmap='magma', alpha=0.8)
    plt.scatter(times, freqs, color='cyan', s=15, edgecolor='black', alpha=0.9, label='Constellation Peaks')
    plt.title('Spectrogram with Constellation Map (Local Maxima)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "constellation_map.png", dpi=300)
    plt.show()

# %% [markdown]
# ## 4. Robustness Experiments
# **Why join two peaks into a hash?**
# A single peak (frequency + time) contains very little information. A noisy environment can easily produce stray frequencies. Joining two peaks into `(freq1, freq2, time_delta)` creates a combinatorially unique hash that encodes structural relationships. Matching structural pairs drops the false-positive rate to near-zero.
# 
# **Noise Addition:**
# White noise broadly elevates the noise floor of the spectrogram. Because our peak extraction uses a local threshold (`maximum_filter`), prominent peaks still survive until the noise fundamentally overwhelms the signal energy (SNR < 0 dB).
# 
# **Pitch Shifting vs Time Stretching:**
# - **Pitch Shift:** Even a slight pitch shift moves ALL frequencies up or down. Because our hash encodes exact absolute frequencies `(f1, f2)`, a pitch shift causes every single hash to mismatch the database, defeating the identifier completely, even if it sounds similar to humans.
# - **Time Stretch:** A time stretch alters the `time_delta` between peaks, which also breaks the hash.
# 
# **Suggested Improvement for Robustness:**
# To make the system robust to pitch shifting, instead of storing absolute frequencies `(f1, f2)`, we could store the frequency *ratio* `f2 / f1` and pitch-shift the `time_delta` proportionally. This makes the fingerprint invariant to linear frequency scaling.
