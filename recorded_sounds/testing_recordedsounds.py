import librosa
import numpy as np
import matplotlib.pyplot as plt
import random

def compare_sound_files(file1, file2, threshold=0.1):
    num = random.randint(1,19)
    # Load sound files
    y1, sr1 = librosa.load(file1)
    y2, sr2 = librosa.load(file2)

   
    # Plot waveforms
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Waveform Comparison')
    plt.plot(y1)
    plt.plot(y2)
    plt.legend(['File 1', 'File 2'])

    # Compute and plot spectrograms
    S1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
    S2 = librosa.feature.melspectrogram(y=y2, sr=sr2)
    plt.subplot(2, 2, 3)
    librosa.display.specshow(librosa.power_to_db(S1, ref=np.max), sr=sr1)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram File 1')
    plt.subplot(2, 2, 4)
    librosa.display.specshow(librosa.power_to_db(S2, ref=np.max), sr=sr2)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram File 2')
    plt.savefig(f'./recorded_sounds//images/trial{num}.png')

    # Extract and compare features
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr1)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr2)
    cosine_distance = np.linalg.norm(mfcc1 - mfcc2)

        # Check if cosine distance is below the threshold
    if cosine_distance < threshold:
        print('The sound files are similar.')
    else:
        print('The sound files are different.')
    print('Cosine distance between MFCCs:', cosine_distance)

    plt.tight_layout()
    plt.show()

# Example usage
file1 = './recorded_sounds/recorded_soun3d.wav'
file2 = './recorded_sounds/recorded_soun4d.wav'
compare_sound_files(file1, file2)
