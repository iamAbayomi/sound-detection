import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import os

# Function to generate the number of the sound file
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

# Function to load data for training LSTM
def load_data_for_lstm(data_dir):
    X = []
    y = []

    # Iterate over each audio file in the no sounds directory
    no_dir = os.path.join(data_dir, "no_sounds")
    for filename in os.listdir(no_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(no_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append([1, 0, 0])  # One-hot encoding for 'no_sound'

    # Iterate over each audio file in the clap_sounds directory
    clap_dir = os.path.join(data_dir, "clap_sounds")
    for filename in os.listdir(clap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(clap_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append([0, 1, 0])  # One-hot encoding for 'clap'

    # Iterate over each audio file in the tap_sounds directory
    tap_dir = os.path.join(data_dir, "tap_sounds")
    for filename in os.listdir(tap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(tap_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append([0, 0, 1])  # One-hot encoding for 'tap'

    return X, y

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: 'no_sound', 'clap', 'tap'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train LSTM model
def train_lstm_model(data_dir):
    # Load data
    X, y = load_data_for_lstm(data_dir)
    X = np.array(X)
    y = np.array(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Reshape data for LSTM input
    input_shape = (X_train.shape[1], X_train.shape[2])
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Create and train LSTM model
    model = create_lstm_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    return model

# Function to classify a new sound with LSTM model
def classify_with_lstm(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    features = np.array(features).reshape((1, len(features), 1))

    # Predict class probabilities using the trained model
    class_probabilities = model.predict(features)[0]

    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(class_probabilities)

    # Define classes
    classes = ['no_sound', 'clap', 'tap']

    # Print the predicted class
    print(f'Predicted class: {classes[predicted_class_index]}')


# Paths to the directory containing training data and the new sound to classify
data_dir = './data/'

# Record a new sound
duration = 5  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

file_path = "rstext.txt"

file_name = './data/recorded_sounds/recorded_sound' + str(generate_num_sound(file_path)) + '.wav'  # Path to save the recorded sound
record_sound(duration, sample_rate, channels, file_name)

# Train the LSTM model
model = train_lstm_model(data_dir)

# Classify the recorded sound using the trained LSTM model
classify_with_lstm(file_name, model)
