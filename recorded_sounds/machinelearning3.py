import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from audiomentations import Compose, PitchShift, TimeStretch

# Function to extract features from audio waveform
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    energy = np.sum(np.abs(y))
    zero_crossings = np.sum(librosa.zero_crossings(y))
    duration = librosa.get_duration(y=y, sr=sr)
    max_amplitude = np.max(np.abs(y))
    return [energy, zero_crossings, duration, max_amplitude]

# Function to extract features from multiple audio files
def extract_features_from_files(audio_files, label):
    features = []
    labels = []
    for file in audio_files:
        file_features = extract_features(file)
        features.append(file_features)
        labels.append(label)
    return features, labels

# Function for data augmentation using pitch shifting and time stretching
def augment_audio(audio_file):
    augmentation = Compose([
        PitchShift(min_semitones=-4, max_semitones=4),
        TimeStretch(min_rate=0.8, max_rate=1.2)
    ])
    y, sr = librosa.load(audio_file)
    augmented_samples = augmentation(samples=y, sample_rate=sr)
    return augmented_samples, sr
