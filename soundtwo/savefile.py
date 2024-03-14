import sounddevice as sd
import soundfile as sf

# Parameters for recording audio
duration = 2  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

# Dictionary to store actions associated with recorded sounds
actions = {
    "tap_tap": "No action",
    "tap_tap_tap": "Browse the internet",
    "tap_tap_tap_tap": "Read the news"
}

# Start recording audio
print(f"Recording for {duration} seconds...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
sd.wait()

# Save the recorded audio to a WAV file
file_name = "recorded_soun4d.wav"
sf.write(file_name, audio_data, sample_rate)

# Write the action associated with the recorded sound to a text file
with open("recorded_actions.txt", "a") as file:
    action = actions.get(input("Enter the action associated with the recorded sound: ").lower(), "Unknown action")
    file.write(f"Sound file: {file_name}, Action: {action}\n")

print(f"Recording saved as {file_name}")
