import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense


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


# Load data for training
def load_data(data_dir):
    X = []
    y = []

    # Iterate over each audio file in the no sounds directory
    no_dir = os.path.join(data_dir, "no_sounds")
    for filename in os.listdir(no_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(no_dir, filename)
            X.append(audio_file)
            y.append("no_sound")

    # Iterate over each audio file in the clap_sounds directory
    clap_dir = os.path.join(data_dir, "clap_sounds")
    for filename in os.listdir(clap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(clap_dir, filename)
            X.append(audio_file)
            y.append("clap")

    # Iterate over each audio file in the tap_sounds directory
    tap_dir = os.path.join(data_dir, "tap_sounds")
    for filename in os.listdir(tap_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(tap_dir, filename)
            X.append(audio_file)
            y.append("tap")

    return X, y


# Function to extract features from audio waveform
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    energy = np.sum(np.abs(y))
    zero_crossings = np.sum(librosa.zero_crossings(y))
    duration = librosa.get_duration(y=y, sr=sr)
    max_amplitude = np.max(np.abs(y))
    return [energy, zero_crossings, duration, max_amplitude]


# Function to preprocess the data
def preprocess_data(X, y):
    # Convert y labels to numerical format
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Extract features for each audio file
    X_features = []
    for audio_file in X:
        features = extract_features(audio_file)
        X_features.append(features)

    # Convert features to numpy array
    X_features = np.array(X_features)

    return X_features, y_encoded


# Train the LSTM model
def train_lstm_model(data_dir):
    # Load data
    X, y = load_data(data_dir)

    # Preprocess the data
    X_features, y_encoded = preprocess_data(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42)

    # Reshape input features to match LSTM input shape
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)

    return model

# Function to classify a new sound and perform corresponding action
# def classify_and_execute(audio_file, model):
#     # Extract features from the new sound
#     features = extract_features(audio_file)
#     # Convert features to numpy array and add timestep dimension
#     features = np.expand_dims(np.array(features), axis=0)
#     features = np.expand_dims(features, axis=1)
#     # Predict class using the trained model
#     prediction = model.predict(features)
#     print("prediction ", prediction)
#     # Decode the prediction
#     predicted_class = "clap" if prediction > 0.5 else "no_sound"

#     # Perform action based on the predicted class
#     if predicted_class == 'clap':
#         print('Clap sound detected! Performing action...')
#         # Code to perform action for clap sound
#     else:
#         print('No clap sound detected.')

# Function to classify a new sound and perform corresponding action
def classify_and_execute(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    # Convert features to numpy array and add timestep dimension
    features = np.expand_dims(np.array(features), axis=0)
    features = np.expand_dims(features, axis=1)
    # Predict class probabilities using the trained model
    probabilities = model.predict(features)
    # Get the predicted class index with the highest probability
    predicted_class_index = np.argmax(probabilities)
    print("prediction ",  predicted_class_index)
    # Decode the predicted class
    if predicted_class_index == 0:
        print('No sound detected!')
        # Code to handle no sound case
    elif predicted_class_index == 1:
        print('Tap sound detected! Performing action...')
        # Code to handle tap sound case
    elif predicted_class_index == 2:
        print('Clap sound detected! Performing action...')
        # Code to handle clap sound case


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

print("model ", model)
# Classify and execute action for the new sound
# Add code to classify the new sound using the trained LSTM model


classify_and_execute(file_name, model)