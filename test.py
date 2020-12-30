from pydub import AudioSegment

newAudio = AudioSegment.from_wav("XC62823.wav")
for i in range(1, 1000000, 200):
    newAudio = newAudio[i:i+10000]
    exportstr = "newsong" + str(i) + ".wav"
    newAudio.export(exportstr, format="wav") #Exports to a wav file in the current path.
