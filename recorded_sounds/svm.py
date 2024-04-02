import os
import librosa
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

# Directory paths for tap and non-tap audio files
tap_dir = './data/recorded_sounds/tap_sounds/'
nontap_dir = './data/no_sounds/'

# Get list of tap and non-tap audio files
tap_files = [os.path.join(tap_dir, file) for file in os.listdir(tap_dir) if file.endswith('.wav')]
nontap_files = [os.path.join(nontap_dir, file) for file in os.listdir(nontap_dir) if file.endswith('.wav')]

# Extract features and labels from tap and non-tap audio files
tap_features, tap_labels = extract_features_from_files(tap_files, 'tap')
nontap_features, nontap_labels = extract_features_from_files(nontap_files, 'nontap')

# Concatenate tap and non-tap features and labels
X = tap_features + nontap_features
y = tap_labels + nontap_labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=22)

# Train the classifier
clf = SVC(probability=True)
clf.fit(X_train, y_train)

# Predict classes for test samples
y_pred = clf.predict(X_test)

# Evaluate classifier performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Threshold for tap detection
threshold = 0.5

# Classify a new audio sample
new_audio_file = './data/recorded_sounds/tap2.wav'
new_features = extract_features(new_audio_file)
prediction = clf.predict([new_features])[0]


print("prediction ", prediction)
print("predict prob ", clf.predict_proba([new_features])[0])

print("predict prob ", clf.predict_proba([new_features])[0][1])

if prediction == 'tap' and clf.predict_proba([new_features])[0][0] > threshold:
    print('Tap sound detected!')
else:
    print('No tap sound detected.')
