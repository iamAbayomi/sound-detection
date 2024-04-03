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
    with open(file_name, "r") as file:
        current_number = int(file.read().strip())
    # Increase the number and add one to it
    num = current_number + 1
    # Write the new number back to the file
    with open(file_name, "w") as file:
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

    # Iterate over each audio file in the no sounds directory
    no_dir = os.path.join(data_dir, "no_sounds")
    for filename in os.listdir(no_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(no_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append("no_sound")

    return X, y

# Train the model
def train_model(data_dir):
    # Load data
    X, y = load_data(data_dir)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Initialize and train model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    return clf

# Function to classify a new sound and count the number of taps or claps
def count_taps_or_claps(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    # Predict class using the trained model
    prediction = model.predict([features])[0]
    # Initialize tap and clap counts
    tap_count = 0
    clap_count = 0
    # Load the audio waveform
    y, sr = librosa.load(audio_file)
    # Calculate onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Define threshold for peak picking
    threshold = 0.5 * np.max(onset_env)
    # Find peaks in onset envelope
    peaks = librosa.util.peak_pick(onset_env, pre_max=20, post_max=20, pre_avg=75, post_avg=75, delta=threshold, wait=20)
    print("#peaks ", peaks, " prediction ", prediction)
    # Iterate over peaks and classify each one
    for peak in peaks:
        # Extract features for the current peak
        peak_features = [energy, zero_crossings, duration, max_amplitude] = extract_features(audio_file)
        # Predict class using the trained model
        peak_prediction = model.predict([peak_features])[0]
        # Increment tap or clap count based on the predicted class
        if peak_prediction == 'tap':
            tap_count += 1
        elif peak_prediction == 'clap':
            clap_count += 1
    return tap_count, clap_count


# Function to classify a new sound and perform corresponding action
def classify_and_execute(audio_file, model):
    # Count taps or claps
    tap_count, clap_count = count_taps_or_claps(audio_file, model)
    # Perform action based on tap and clap counts
    if tap_count > 0:
        print(f'{tap_count} tap(s) detected! Performing action for taps...')
        # Code to perform action for tap sound (e.g., open application)
    if clap_count > 0:
        print(f'{clap_count} clap(s) detected! Performing action for claps...')
        # Code to perform action for clap sound
    if tap_count == 0 and clap_count == 0:
        print('No tap or clap sound detected.')


# Paths to the directory containing training data and the new sound to classify
data_dir = './data/'

# Record a new sound
duration = 5  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

file_path = "rstext.txt"

file_name = './data/recorded_sounds/recorded_sound' + str(generate_num_sound(file_path)) +  '.wav'  # Path to save the recorded sound
record_sound(duration, sample_rate, channels, file_name)

# Train the model
model = train_model(data_dir)

# Classify and execute action for the new sound
classify_and_execute(file_name, model)

