import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

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

    # Iterate over each audio file in the no sounds directory
    no_dir = os.path.join(data_dir, "no_sounds")
    for filename in os.listdir(no_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(no_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append("no_sound")

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

    return np.array(X), np.array(y)

# Define LSTM architecture
def create_lstm(input_shape, num_classes):
    model = Sequential([
        LSTM(128, input_shape=input_shape),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Train the LSTM model
def train_lstm_model(X_train, y_train, input_shape, num_classes):
    model = create_lstm(input_shape, num_classes)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
    return model

# Function to classify a new sound and perform corresponding action
def classify_and_execute(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    # Reshape features to match LSTM input shape
    features = features.reshape(1, features.shape[0], 1)
    # Predict class using the trained model
    prediction = np.argmax(model.predict(features))
    print("prediction ", prediction)
    # Perform action based on the predicted class
    if prediction == 0:
        print('No sound detected.')
    elif prediction == 1:
        print('Clap sound detected! Performing action...')
        # Code to perform action for clap sound
    elif prediction == 2:
        print('Tap sound detected! Performing action...')
        # Code to perform action for tap sound

# Paths to the directory containing training data and the new sound to classify
data_dir = './data/'

# Record a new sound
duration = 5  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio
file_name = './data/recorded_sounds/recorded_sound.wav'  # Path to save the recorded sound
record_sound(duration, sample_rate, channels, file_name)

# Load training data
X, y = load_data(data_dir)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the LSTM model
input_shape = (X_train.shape[1], 1)  # Adjusted input shape
num_classes = len(np.unique(y))
model = train_lstm_model(X_train, y_train, input_shape, num_classes)

# Classify and execute action for the new sound
classify_and_execute(file_name, model)

