import sounddevice as sd
import numpy as np
import queue

# Parameters for recording audio
duration = 5  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

# Queue to store audio chunks
q = queue.Queue()

# Callback function to capture audio chunks
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# Start recording audio
with sd.InputStream(callback=callback, channels=channels, samplerate=sample_rate):
    print(f"Recording for {duration} seconds...")
    sd.sleep(int(duration * 1000))

# Combine all recorded chunks into a single numpy array
audio_data = np.concatenate(list(q.queue), axis=0)

# Play back the recorded sound
print("Playing back the recorded sound...")
sd.play(audio_data, sample_rate)
sd.wait()  # Wait until playback is finished
print("Playback finished.")
