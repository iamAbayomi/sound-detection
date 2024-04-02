import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Sample tap and non-tap audio files
tap_file = './data/recorded_sounds/tap.wav'
nontap_file = './data/recorded_sounds/nosound.wav'

# Extract features from tap and non-tap audio files
tap_features = extract_features(tap_file)
nontap_features = extract_features(nontap_file)

# Train a simple classifier (Random Forest) on the extracted features
X = [tap_features, nontap_features]
y = ['tap', 'nontap']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict classes for test samples
y_pred = clf.predict(X_test)

print(f'tap_features', tap_features, 'nontap_features',nontap_features ,'ypred', y_pred)

# Evaluate classifier performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Threshold for tap detection
threshold = 0.5

# Classify a new audio sample
new_audio_file = './data/recorded_sounds/lala.wav'
new_features = extract_features(new_audio_file)
prediction = clf.predict([new_features])[0]


print(f'tap_features', tap_features, 'nontap_features',nontap_features ,'ypred', y_pred, ' new_features ', new_features)

print("predict prob ", clf.predict_proba([new_features]))

if prediction == 'tap' and clf.predict_proba([new_features])[0][0] > threshold:
    print('Tap sound detected!')
else:
    print('No tap sound detected.')
