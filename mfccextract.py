import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load('C:/sound/XC62823.wav')

print(y.shape)
segmentation = 100000

for i in range(1, len(y)):
    if(i%segmentation==0):
        newy = y[i-segmentation:i]
        newy = librosa.feature.mfcc(newy)
        plt.figure()
        librosa.display.specshow(newy, sr=sr)
        plt.tight_layout()

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