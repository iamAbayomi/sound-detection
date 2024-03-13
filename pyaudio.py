import pyaudio
import numpy as np

# Constants
CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1  # Number of audio channels (mono)
RATE = 44100  # Sampling rate
THRESHOLD = 2000  # Adjust this threshold based on your environment and microphone sensitivity

# Function to detect claps
def detect_claps(data):
    # Convert byte data to numpy array
    audio_data = np.frombuffer(data, dtype=np.int16)
    # Calculate root mean square (RMS) amplitude
    rms = np.sqrt(np.mean(np.square(audio_data)))
    # Check if RMS amplitude exceeds threshold
    if rms > THRESHOLD:
        return True
    return False

# Callback function for audio stream
def callback(in_data, frame_count, time_info, status):
    if detect_claps(in_data):
        print("Clap detected!")
    return (in_data, pyaudio.paContinue)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open audio stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=callback
)

# Start audio stream
stream.start_stream()

# Wait for stream to finish
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    pass

# Stop audio stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
audio.terminate()
