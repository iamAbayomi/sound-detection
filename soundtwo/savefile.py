import sounddevice as sd
import soundfile as sf
import os  # Import the os module to work with file paths
from random import randint

# Parameters for recording audio
duration = 2  # Duration of recording in seconds
sample_rate = 44100  # Sampling rate
channels = 1  # Mono audio

# num = randint(1,100)

file_path = "num.txt"

# Read the current number from the file
with open(file_path, "r") as file:
    current_number = int(file.read().strip())

# Increase the number and add one to it
num = current_number + 1



# Specify the directory where you want to save the recorded file
save_dir = "./data/random_sounds/" 

# Ensure the specified directory exists, create it if necessary
os.makedirs(save_dir, exist_ok=True)

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

name = "random" + str(num) + ".wav"
# Construct the full file path including the directory and file name
file_name = os.path.join(save_dir, name)

sf.write(file_name, audio_data, sample_rate)



# Write the new number back to the file
with open(file_path, "w") as file:
      file.write(str(num))

# # Write the action associated with the recorded sound to a text file
# with open("recorded_actions.txt", "a") as file:
#     action = actions.get(input("Enter the action associated with the recorded sound: ").lower(), "Unknown action")
#     file.write(f"Sound file: {file_name}, Action: {action}\n")

print(f"Recording saved as {file_name}")
