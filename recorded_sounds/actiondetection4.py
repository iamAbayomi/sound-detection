import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import webbrowser
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def generate_num_sound(file_path):
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

    max_pos = np.argmax(y)
    # 
    x_chunk = y[max_pos-512:max_pos+3584]

    n_fft, hop_length,win_length  = 512, 256, 512

    mfcc = librosa.feature.mfcc(y=x_chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length, n_mfcc=13)
    
    # energy = np.sum(np.abs(y))
    # zero_crossings = np.sum(librosa.zero_crossings(y))
    # duration = librosa.get_duration(y=y, sr=sr)
    # max_amplitude = np.max(np.abs(y))
    return mfcc.flatten() #[energy, zero_crossings, duration, max_amplitude]

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

    # Iterate over each audio file in the no_sounds directory
    no_dir = os.path.join(data_dir, "no_sounds")
    for filename in os.listdir(no_dir):
        if filename.endswith(".wav"):
            audio_file = os.path.join(no_dir, filename)
            features = extract_features(audio_file)
            X.append(features)
            y.append("no_sound")

    return X, y

# Function to classify a new sound and count the number of snaps or claps
def count_snaps_or_claps(audio_file, model):
    # Extract features from the new sound
    features = extract_features(audio_file)
    # Predict class using the trained model
    prediction = model.predict([features])[0]
    # Initialize snap and clap counts
    snap_count = 0
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

        # peak_features = extract_features(audio_file)
        # Predict class using the trained model
        peak_prediction = model.predict([peak_features])[0]
        # Increment snap or clap count based on the predicted class
        if peak_prediction == 'snap':
            snap_count += 1
        elif peak_prediction == 'clap':
            clap_count += 1
    return snap_count, clap_count



# Train the model
def train_model(data_dir):
    # Load data
    X, y = load_data(data_dir)

    print(" X ", X)
    print(" y ", y)

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

def determineActionForClaps(clap_count):
    if clap_count == 1:
        # Code to call emergency number
        print("Calling Emergency Number")
        emergency_number = "tel:911"  # Change this to your local emergency number
        webbrowser.open(emergency_number)
    elif clap_count == 2:
        # Call to open the browser
        webbrowser.open('https://www.google.com')
    elif clap_count == 3:
        # Code to read the daily news
        webbrowser.open('https://www.independent.ie')
    elif clap_count == 4:
        # Command to open the screen reader on macOS
        screen_reader_command = "open -a VoiceOver"
        # Execute the command to open the screen reader
        subprocess.Popen(screen_reader_command, shell=True)
    elif clap_count == 5:
        # Command to open the default music player on macOS
        music_player_command = "open -a Music"
        # Execute the command to open the default music player
        subprocess.Popen(music_player_command, shell=True)
        pass


def determineActionForsnaps(snap_count):
    if snap_count == 1:
        print("Open Siri")
        # Open Siri
        subprocess.run(['open', '-a', 'Siri'])
    elif snap_count == 2:
        print("Open email client")
        # Open default email client
        subprocess.run(['open', 'mailto:'])
    elif snap_count == 3:
        print("Take a screenshot")
        # Take a screenshot
        subprocess.run(['screencapture', 'screenshot.png'])
    elif snap_count == 4:
        print("Open the calendar")
        # Open Calendar app
        subprocess.run(['open', '-a', 'Calendar'])
    elif snap_count == 5:
        print("Lock the computer")
        # Lock the computer
        subprocess.run(['pmset', 'displaysleepnow'])



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

