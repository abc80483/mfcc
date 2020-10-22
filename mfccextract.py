import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import sklearn

y, sr = librosa.load('C:/sound/XC62823.wav')

print(y.shape)
move = 20000
segmentation = 50000

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

"""for i in range(1, len(y), move):
    newy = y[i:i+segmentation]
    newy = librosa.feature.mfcc(newy, n_mels=64, fmax=8000)
    plt.figure(figsize=(20,5))
    librosa.display.specshow(newy, sr=sr, y_axis='mel', x_axis='time', fmax=8000)
    plt.tight_layout()
    plt.show()"""

y = normalize_audio(y)

for i in range(1, len(y), move):
    if(i+segmentation<y.shape[0]):
        newy = y[i:i + segmentation]
        mfcc = librosa.feature.mfcc(newy, n_mels=40, fmax=80000)
        newmfcc = sklearn.preprocessing.scale(mfcc, axis=1)
        plt.figure(figsize=(5, 4))
        plt.plot(np.linspace(0, len(newy) / sr, num=len(newy)), newy)

        sum = 0
        for j in newy:
            sum += abs(j)

        threshold = sum / newy.shape[0]
        plotthreshold = np.zeros((50000,1))
        plotthreshold[,:] = threshold*3
        print(plotthreshold)
        #plt.axhline(y=plotthreshold, linewidth=1, color='k')
        #plt.plot(np.linspace(0, len(newy) / sr, num=len(newy)), plotthreshold, 'r-')
        plt.grid(True)
        plt.show()
        """plt.figure(figsize=(5, 4))
        librosa.display.specshow(mfcc, sr=sr, y_axis='mel', x_axis='time')
        plt.axis('off')
        plt.show()"""
"""        
y = librosa.feature.mfcc(y)
np.swapaxes(y, 1, 0)
librosa.display.specshow(y, sr=sr, )

plt.colorbar(format='%+2.0f dB')
plt.figure()
plt.plot(y)
plt.title("mfcc")
plt.show()
"""