import soundfile as sf
import sounddevice as sd
import numpy as np

# Load the recorded sound file
file_name = "recorded_sound.wav"
recorded_data, sample_rate = sf.read(file_name)

# Record a new sound using the code you provided
duration = 5  # Duration of recording in seconds
new_recorded_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()

# Compare the audio data
if np.array_equal(recorded_data, new_recorded_data):
    print("The recorded sound matches the sound in the file.")
else:
    print("The recorded sound does not match the sound in the file.")
