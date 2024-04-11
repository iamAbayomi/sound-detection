import librosa
import numpy as np
import matplotlib.pyplot as plt
import random


def display_sound(file):
    filepath = file.split('/')[3]
    filename = filepath.split('.')[0]
    print(file.split('/')[3])
    # Load sound files
    y1, sr1 = librosa.load(file)

    # Plot waveforms
    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.title('Waveform Details')
    plt.plot(y1)

    # Compute and plot spectrogram
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1, hop_length=128, win_length=256)
    plt.subplot(2,1,2)
    # plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S1, ref=np.max), sr=sr1)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(f'./data/images/{filename}.png')
   

    plt.tight_layout()
    plt.show()



# Clap sounds
# file = './data/clap_sounds/clap5.wav'

# Tap sounds
file = './data/tap_sounds/tap2.wav'

display_sound(file)