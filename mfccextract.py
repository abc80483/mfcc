import librosa
import matplotlib as plt

y, sr = librosa.load('C:/soundfolder/XC62823.wav', sr=None)
#plot
plt.figure()
librosa.display.waveplot(y,sr)
plt.title("test")
plt.show()
