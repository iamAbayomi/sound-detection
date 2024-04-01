import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Define paths to your audio data folders (replace with your actual paths)
tap_data_path = './Users/oladiniabayomi/Documents/Documents/Masters /2nd Semester/Engineering Team Project/code/project/data/recorded_sounds/tap.wav'
noise_data_path = './data/recorded_sounds/nosound.wav'

# Function to load audio data and extract MFCC features
def load_data(data_path, label):
  features = []
  for filename in os.listdir(data_path):
    # Load audio
    y, sr = librosa.load(os.path.join(data_path, filename))
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    # Flatten features for machine learning model
    features.append(mfccs.flatten())
  return np.array(features), np.array([label] * len(features))

# Load tap and noise data, extract features, and create labels
tap_features, tap_labels = load_data(tap_data_path, 1)
noise_features, noise_labels = load_data(noise_data_path, 0)

# Combine features and labels
all_features = np.concatenate((tap_features, noise_features))
all_labels = np.concatenate((tap_labels, noise_labels))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2)

# Train a Support Vector Machine (SVM) model
model = SVC()
model.fit(X_train, y_train)

# Evaluate on testing set (replace with your own evaluation metrics)
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# Function to predict if a new audio clip is a tap
def predict_tap(audio_path):
  # Load audio, extract features
  y, sr = librosa.load(audio_path)
  mfccs = librosa.feature.mfcc(y=y, sr=sr)
  test_feature = np.array([mfccs.flatten()])
  # Predict using the trained model
  prediction = model.predict(test_feature)
  return "Tap" if prediction[0] == 1 else "Not Tap"

# Example usage (replace with your audio file path)
audio_file = './data/recorded_sounds/tap2.wav'
prediction = predict_tap(audio_file)
print("Prediction:", prediction)
