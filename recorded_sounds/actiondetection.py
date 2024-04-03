import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def generate_num_sound(file_name):
    # Generate file name 
    with open(file_path, "r") as file:
        current_number = int(file.read().strip())
    # Increase the number and add one to it
    num = current_number + 1
    # Write the new number back to the file
    with open(file_path, "w") as file:
        file.write(str(num))
    return num

# Function to record sound
def record_sound(duration, sample_rate, channels, file_name):
    print(f"Recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    # Save the recorded audio to a WAV file
    sf.write(file_name, audio_data, sample_rate)
    print(f"Recording saved as {file_name}")

# Function to extract features from audio waveform
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    energy = np.sum(np.abs(y))
    zero_crossings = np.sum(librosa.zero_crossings(y))
    duration = librosa.get_duration(y=y, sr=sr)
    max_amplitude = np.max(np.abs(y))
    return [energy, zero_crossings, duration, max_amplitude]

# Load data for training
def load_data(data_dir):
    X = []
    y = []

    # Iterate over each audio file in the clap_sounds directory
    clap_dir = os.path.join(data_dir, "clap_sounds")
    for filename in os.listdir(clap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(clap_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append("clap")

    # Iterate over each audio file in the tap_sounds directory
    tap_dir = os.path.join(data_dir, "tap_sounds")
    for filename in os.listdir(tap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(tap_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append("tap")

    return X, y

# Train the model
def train_model(data_dir):
    # Load data
    X, y = load_data(data_dir)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    return clf

# Function to classify a new sound and perform corresponding action
def classify_and_execute(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    # Predict class using the trained model
    prediction = model.predict([features])[0]
    # Perform action based on the predicted class
    if prediction == 'tap':
        print('Tap sound detected! Performing action...')
        # Code to perform action for tap sound (e.g., open application)
    elif prediction == 'clap':
        print('Clap sound detected! Performing action...')
        # Code to perform action for clap sound
    else:
        print('Unknown sound detected.')


# Paths to the directory containing training data and the new sound to classify
data_dir = './data/'

# Record a new sound
duration = 2  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

file_path = "rstext.txt"

file_name = './data/recorded_sounds/recorded_sound' + str(generate_num_sound(file_path)) +  '.wav'  # Path to save the recorded sound
record_sound(duration, sample_rate, channels, file_name)

# Train the model
model = train_model(data_dir)

# Classify and execute action for the new sound
classify_and_execute(file_name, model)

