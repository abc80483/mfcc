import librosa
import matplotlib.pyplot as plt

y, sr = librosa.load('C:/sound/XC62823.wav')
#plot
plt.figure()
plt.plot(y)
plt.title("test")
plt.show()

y = librosa.feature.mfcc(y)

plt.figure()
plt.plot(y)
plt.title("mfcc")
plt.show()
