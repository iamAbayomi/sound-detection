import librosa
import numpy as np

def extract_features(audio_file):
    """
    Extracts features from audio waveform.
    
    Parameters:
    - audio_file: Path to the audio file.
    
    Returns:
    - features: List of extracted features.
    """
    # Load audio waveform
    y, sr = librosa.load(audio_file)

    # Extract features
    # Example features:
    # 1. Energy: Total energy of the signal
    energy = np.sum(np.abs(y))

    # 2. Zero Crossings: Number of times the signal crosses the zero axis
    zero_crossings = np.sum(librosa.zero_crossings(y))

    # 3. Spectral Centroid: Center of mass of the spectrum
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # 4. Spectral Roll-off: Frequency below which a certain percentage of the total spectral energy lies
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 5. Mel-frequency Cepstral Coefficients (MFCCs): Representation of the short-term power spectrum of a sound
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    # As an example, we'll return a dictionary with these features
    features = {
        "energy": energy,
        "zero_crossings": zero_crossings,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "mfccs": mfccs
    }

    return features
