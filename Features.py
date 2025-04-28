import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import librosa.display
from matplotlib.animation import FuncAnimation
import gc


'''
what is mfcc?
MFCC stands for Mel-Frequency Cepstral Coefficients. It is a feature extraction technique widely used in speech and audio processing. 
MFCCs capture the shape of the spectral envelope of a sound, which is useful for tasks like speech recognition, music classification, and audio analysis.

What does it mean?
Each MFCC represents different characteristics of the audio:

Lower MFCC coefficients (e.g., 1-12) capture general spectral shape (like loudness and formants).
Higher MFCC coefficients capture fine spectral details (like noise and texture).
MFCCs are commonly used in machine learning models to classify and analyze audio data.

what is n_mfcc?
n_mfcc is the number of mfccs to return
'''


def get_mfcc(y, sr,viz):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=56)
    #print("MFCC shape (coefficients x frames):", mfccs.shape)

    # Normalize MFCCs
    scaler = MinMaxScaler()
    mfccs = scaler.fit_transform(mfccs.T).T

    # Round for readability
    mfccs = np.round(mfccs, 2)

    # --- 1. Heatmap ---
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    plt.colorbar(label="MFCC Coefficients")
    plt.title("MFCC Heatmap")
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.savefig('IMAGES/mfcc_heatmap.png' , dpi=100, bbox_inches='tight')
    
        
    plt.close('all'); gc.collect()
        

    # --- 2. t-SNE Plot ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    mfccs_tsne = tsne.fit_transform(mfccs.T)

    plt.figure(figsize=(8, 6))
    plt.scatter(mfccs_tsne[:, 0], mfccs_tsne[:, 1], s=15, alpha=0.7)
    plt.title("t-SNE Projection of MFCCs")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('IMAGES/tsne_mfcc.png' , dpi=100, bbox_inches='tight')

        
    plt.close('all'); gc.collect()
        

    # --- 3. PCA Plot ---
    pca = PCA(n_components=2)
    mfccs_pca = pca.fit_transform(mfccs.T)

    plt.figure(figsize=(8, 6))
    plt.scatter(mfccs_pca[:, 0], mfccs_pca[:, 1], s=15, alpha=0.7)
    plt.title("PCA Projection of MFCCs")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('IMAGES/pca_mfcc.png' , dpi=100, bbox_inches='tight')

        
    plt.close('all'); gc.collect()


    # --- 4. 3D Surface Plot ---
    X = np.arange(mfccs.shape[1])  # time frames
    Y = np.arange(mfccs.shape[0])  # MFCC indices
    X, Y = np.meshgrid(X, Y)
    Z = mfccs

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('3D MFCC Surface Plot')
    ax.set_xlabel('Time Frame')
    ax.set_ylabel('MFCC Coeff')
    ax.set_zlabel('Value')
    plt.tight_layout()
    plt.savefig('IMAGES/mfcc_surface.png' , dpi=100, bbox_inches='tight')

        
    plt.close('all'); gc.collect()

    return mfccs



'''
what is chroma?
Chroma features represent pitch class energy (C, C#, D, ... B) over time —
it's essentially the "color" of the harmony in music.
'''
def get_chroma(y, sr, viz):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    #print("Chroma shape:", chroma.shape)

    # Normalize Chroma
    scaler = MinMaxScaler()
    chroma = scaler.fit_transform(chroma.T).T
    chroma = np.round(chroma, 2)

    # --- Visualization 1: Traditional Chroma Spectrogram ---
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.colorbar(label='Chroma Intensity')
    plt.title('Chroma Spectrogram')
    plt.tight_layout()
    plt.savefig('IMAGES/chroma_spectrogram.png' , dpi=100, bbox_inches='tight')

        
    plt.close('all'); gc.collect()

    # --- Visualization 2: Polar Tonal Radar with Labels ---
    def animate_chroma_polar(chroma):
        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot(111, polar=True)

        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                         'F#', 'G', 'G#', 'A', 'A#', 'B']
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)

        bars = ax.bar(angles, chroma[:,0], 
                      width=2*np.pi/12, bottom=0.0, alpha=0.8)

        # Add pitch labels
        for i, angle in enumerate(angles):
            ax.text(angle, 1.1, pitch_classes[i], 
                    ha='center', va='center', fontsize=12, weight='bold')

        ax.set_yticklabels([])
        ax.set_xticks([])

        def update(frame):
            for bar, height in zip(bars, chroma[:, frame]):
                bar.set_height(height)
            return bars

        ani = FuncAnimation(fig, update, frames=chroma.shape[1], interval=100, blit=True)
        plt.title("Tonal Radar (Chroma Polar View)")
        plt.tight_layout()

            
        plt.close('all'); gc.collect()
        ani.save('IMAGES/chroma_polar.gif', writer='imagemagick', fps=10)

    animate_chroma_polar(chroma)

    # --- Visualization 3: Vivid Color Stripe Harmony Flow ---
    # Use dark, saturated colors for each pitch class
    vivid_colors = np.array([
        [0.8, 0.1, 0.1],   # C - Red
        [0.9, 0.5, 0.1],   # C#
        [1.0, 0.8, 0.1],   # D
        [0.6, 1.0, 0.2],   # D#
        [0.1, 0.9, 0.1],   # E - Green
        [0.1, 0.8, 0.6],   # F
        [0.1, 0.6, 0.9],   # F#
        [0.1, 0.3, 1.0],   # G - Blue
        [0.3, 0.1, 0.8],   # G#
        [0.5, 0.1, 0.9],   # A
        [0.8, 0.1, 0.6],   # A#
        [0.6, 0.1, 0.4]    # B - Magenta
    ])

    stripe = np.zeros((50, chroma.shape[1], 3))

    for i in range(12):
        for t in range(chroma.shape[1]):
            stripe[:, t, :] += chroma[i, t] * vivid_colors[i]

    stripe = np.clip(stripe, 0, 1)

    plt.figure(figsize=(12, 2))
    plt.imshow(stripe, aspect='auto')
    plt.title("Color Stripe Harmony Flow (Vivid)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('IMAGES/chroma_stripe.png' , dpi=100, bbox_inches='tight')
        
    plt.close('all'); gc.collect()

    return chroma


'''
what is spectral contrast?
Spectral contrast is a feature extraction technique used in audio signal processing.
It measures the difference between peaks and valleys in the audio spectrum.
'''

def get_spectral_contrast(y,sr, viz):
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    #print(spectral_contrast.shape)


    # Normalize Spectral Contrast
    scaler = MinMaxScaler()
    spectral_contrast = scaler.fit_transform(spectral_contrast.T).T
    # Round Spectral Contrast to 2 decimal places
    spectral_contrast = np.round(spectral_contrast, 2)
    #print(spectral_contrast)


    # Plot
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(spectral_contrast, x_axis='time', sr=sr)
    plt.colorbar(label="Normalized Spectral Contrast")
    plt.title("Spectral Contrast Visualization")
    plt.xlabel("Time")
    plt.ylabel("Frequency Band")
    plt.tight_layout()
    plt.savefig('IMAGES/spectral_contrast.png' , dpi=100, bbox_inches='tight')
    
        
    plt.close('all'); gc.collect()
    return spectral_contrast



'''
what is tonnetz?
Tonnetz is a feature extraction technique used in audio signal processing.
It captures the harmonic content of the audio signal.
It is based on the Tonnetz representation of music theory, which maps musical intervals to geometric shapes.
Tonnetz is useful for tasks like music genre classification and chord recognition.
'''
def get_tonnetz(y,sr, viz):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    #print(tonnetz.shape)


    # Normalize Tonnetz
    scaler = MinMaxScaler()
    tonnetz = scaler.fit_transform(tonnetz.T).T

    # Round Tonnetz to 2 decimal places
    tonnetz = np.round(tonnetz, 2)
    #print(tonnetz)


    plt.figure(figsize=(12, 6))
    librosa.display.specshow(tonnetz, x_axis="time", sr=sr)
    plt.colorbar(label="Tonnetz")
    plt.title("Tonnetz Visualization")
    plt.xlabel("Time")
    plt.ylabel("Tonnetz")
    plt.savefig('IMAGES/tonnetz.png' , dpi=100, bbox_inches='tight')
    
        
    plt.close('all'); gc.collect()

    return tonnetz


'''
what is ondset detection?
Onset detection is a feature extraction technique used in audio signal processing.
It detects the beginning of musical notes in an audio signal.
Onset detection is useful for tasks like beat tracking, tempo estimation, and music transcription.
'''

#Get Onset Detection
def get_onset(y,sr, viz):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env = librosa.util.normalize(onset_env)


    # Normalize Onset Envelope
    scaler = MinMaxScaler()
    onset_env = scaler.fit_transform(onset_env.reshape(-1, 1)).flatten()
    #print(onset_env.shape)

    # Round Onset Envelope to 2 decimal places
    onset_env = np.round(onset_env, 2)
    #print(onset_env)

    plt.figure(figsize=(12, 6))
    plt.plot(onset_env)
    plt.title("Onset Detection")
    plt.xlabel("Time")
    plt.ylabel("Onset Strength")
    plt.savefig('IMAGES/onset_detection.png' , dpi=100, bbox_inches='tight')
    
        
    plt.close('all'); gc.collect()
    return onset_env




def get_beat_tempo_tracking(y,sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    #print(tempo)
    #print(beats)
    
    return tempo, beats




#get Mel Spectrogram
def get_mel_spectrogram(y, sr, viz):
    # Extract mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to dB (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize for ML (0 to 1)
    scaler = MinMaxScaler()
    mel_spec_norm = scaler.fit_transform(mel_spec_db.T).T  # Normalize across time axis

    # Round for readability (optional)
    mel_spec_norm = np.round(mel_spec_norm, 2)

    # Plot
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram (dB)')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    # Save before showing
    plt.savefig("IMAGES/mel_spectrogram.png", dpi=100, bbox_inches='tight')
    
        
    plt.close('all'); gc.collect()
    #print("Mel Spectrogram shape:", mel_spec_norm.shape)
    return mel_spec_norm




'''
what does get_tension_score do?
It calculates the tension score of an audio file.
The tension score is a measure of how tense or suspenseful the audio is.
It is computed using a weighted sum of audio features like spectral centroid, spectral bandwidth, RMS energy, zero crossing rate, and tempo.
The tension score is normalized to a range of 0-10.

What is spectral centroid?
Spectral centroid is a feature that represents the center of mass of the audio spectrum.
It indicates the "brightness" of the audio signal, with higher values corresponding to brighter sounds.
Spectral centroid is useful for tasks like timbre analysis and audio classification.

what is spectral bandwidth?
Spectral bandwidth is a feature that measures the width of the audio spectrum.
It indicates the spread of frequencies in the audio signal, with higher values corresponding to wider spectra.
Spectral bandwidth is useful for tasks like instrument recognition and sound source localization.

what is rms energy?
RMS energy is a feature that measures the energy of the audio signal.
It indicates the overall loudness of the audio, with higher values corresponding to louder sounds.
RMS energy is useful for tasks like audio compression and volume normalization.

what is zero crossing rate?
Zero crossing rate is a feature that measures the number of times the audio signal crosses the zero axis.
It indicates the amount of change in the audio signal, with higher values corresponding to more dynamic sounds.
Zero crossing rate is useful for tasks like pitch detection and speech recognition.
'''




def get_tension_score(y, sr, viz):
    # Extract features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    rms_energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Estimate tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    #PLOT spectral centroid
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
    plt.title("Spectral Centroid")
    plt.xlabel("Time")
    
        
    plt.close('all'); gc.collect()

    #plot spectral bandwidth
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
    plt.title("Spectral Bandwidth")
    plt.xlabel("Time")
    
        
    plt.close('all'); gc.collect()

    #plot rms energy
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.feature.rms(y=y)[0])
    plt.title("RMS Energy")
    plt.xlabel("Time")
    
        
    plt.close('all'); gc.collect()

    #plot zero crossing rate
    plt.figure(figsize=(12, 6))
    plt.plot(librosa.feature.zero_crossing_rate(y)[0])
    plt.title("Zero Crossing Rate")
    plt.xlabel("Time")
    
        
    plt.close('all'); gc.collect()


    # Normalize features
    norm_centroid = min(1, spectral_centroid / 5000)
    norm_bandwidth = min(1, spectral_bandwidth / 3000)
    norm_rms = min(1, rms_energy * 100)
    norm_zcr = min(1, zcr * 100)
    norm_tempo = min(1, tempo / 200)

    # Compute tension score
    tension_score = (0.3 * norm_centroid + 
                     0.2 * norm_bandwidth + 
                     0.2 * norm_rms + 
                     0.15 * norm_zcr + 
                     0.15 * norm_tempo) * 10

    return round(tension_score.item(), 2) 


def compute_audio_energy(y, sr, viz):
    # Compute RMS energy
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    
    # Compute Spectral Flux
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    avg_flux = np.mean(onset_env)
    
    # Compute Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Compute Net Energy Score
    energy_score = np.clip((avg_rms * 50 + avg_flux * 30 + tempo * 0.2), 1, 100)
    
    #Plot Energy Levels
    plt.figure(figsize=(10, 4))
    plt.plot(rms, label='RMS Energy', color='r')
    plt.plot(onset_env / np.max(onset_env) * np.max(rms), label='Spectral Flux (Scaled)', color='b')
    plt.axhline(y=avg_rms, color='r', linestyle='--', label='Avg RMS')
    plt.axhline(y=avg_flux, color='b', linestyle='--', label='Avg Flux')
    plt.legend()
    plt.title('MEL Spectrogram and Energy Levels')
    plt.xlabel('Frames')
    plt.ylabel('Amplitude')
    
        
    plt.close('all'); gc.collect()
    
    return round(energy_score.item(), 2)



def get_three_segment_features(feature_2d, segment_frames=200):
    n_frames = feature_2d.shape[1]
    total_required = segment_frames * 3

    # Pad if too short
    if n_frames < total_required:
        pad_width = total_required - n_frames
        feature_2d = np.pad(feature_2d, ((0, 0), (0, pad_width)), mode='constant')
        n_frames = feature_2d.shape[1]

    thirds = n_frames // 3

    start = feature_2d[:, 0 : segment_frames]
    middle = feature_2d[:, thirds : thirds + segment_frames]
    end = feature_2d[:, 2 * thirds : 2 * thirds + segment_frames]

    return np.concatenate([start, middle, end], axis=1)  # shape: (features, 600)



def visualize_segment_windows(y, sr, segment_frames=200):
    hop_length = 512  # default
    total_frames = int(len(y) / hop_length)
    duration = librosa.get_duration(y=y, sr=sr)

    time_per_frame = duration / total_frames
    window_sec = segment_frames * time_per_frame

    thirds = total_frames // 3
    times = np.linspace(0, duration, total_frames)

    # Get start times of each segment in seconds
    start_time = 0
    middle_time = thirds * time_per_frame
    end_time = 2 * thirds * time_per_frame

    # Plot the waveform
    plt.figure(figsize=(14, 3))
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    
    # Overlay shaded segments
    for label, start in zip(["Start", "Middle", "End"], [start_time, middle_time, end_time]):
        plt.axvspan(start, start + window_sec, alpha=0.3, label=f"{label} Segment")

    plt.title("Segment Windows on Audio Timeline")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    
    plt.close('all'); gc.collect()


def compile_features(audio_path, segment_frames=200, viz=False, SEGMENTS = False):
    y, sr = librosa.load(audio_path, sr=None)

    # --- Extract and Segment Frame-based Features (each becomes shape: (n, 600)) ---
    mfcc = get_three_segment_features(get_mfcc(y, sr, viz), segment_frames)
    chroma = get_three_segment_features(get_chroma(y, sr,viz), segment_frames)
    spectral_contrast = get_three_segment_features(get_spectral_contrast(y, sr,viz), segment_frames)
    tonnetz = get_three_segment_features(get_tonnetz(y, sr,viz), segment_frames)
    mel_spec = get_three_segment_features(get_mel_spectrogram(y, sr,viz), segment_frames)

    # --- Flatten All to 1D ---
    flat_mfcc = mfcc.flatten()
    flat_chroma = chroma.flatten() 
    flat_contrast = spectral_contrast.flatten()
    flat_tonnetz = tonnetz.flatten()
    flat_mel = mel_spec.flatten()

    # --- Onset: Flatten 600-frame signal ---
    onset = get_onset(y, sr,viz)
    onset = np.pad(onset, (0, max(0, segment_frames * 3 - len(onset))), mode='constant')[:segment_frames * 3]

    # --- Combine All Features ---
    features = np.concatenate([
        flat_mfcc,
        flat_chroma,
        flat_contrast,
        flat_tonnetz,
        flat_mel,
        onset
    ])

    # --- Scalar Features ---
    tension_score = get_tension_score(y, sr, viz)
    energy_score = compute_audio_energy(y, sr, viz)
    BPM = get_beat_tempo_tracking(y, sr)[0]

    # --- Normalize Combined Feature Vector ---
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features.reshape(-1, 1)).flatten()

    #print("Compiled feature vector shape:", features.shape)

    if SEGMENTS == True:
        visualize_segment_windows(y, sr, segment_frames=segment_frames)
        #print("Segment frames:", segment_frames)


    return features, tension_score, energy_score, BPM




'''
Next steps

Design model that uses 3 values Features array images, and Scaler values to get a neural network to cluster the songs into different groups.

then use the saved model to predict the cluster of a new song.

Look at the t-sne plot and see if the clusters are visible.

'''
