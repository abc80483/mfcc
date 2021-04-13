import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import librosa
import librosa.display
import os
import time
from sklearn.preprocessing import scale
import struct
import copy
import keyboard


CHUNK = 1024 * 1
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

Fs = 22050

def plot_spec(spec, filename):
    plt.clf()

    #spec = scale(spec, axis=1)#********************************
    #cmap = LinearSegmentedColormap.from_list("", ["#007979", "#00A600", "#9AFF02", "#FFFFFF", "#FF0000"])
    cmap = LinearSegmentedColormap.from_list("", ["#FFFFFF", "#FFFFFF", "000000", "#000000", "#000000"])

    """if np.average(spec) > 0:
        norm = plt.Normalize(-5, 40)
    elif np.average(spec) < 0 and np.average(spec) > -8:
        norm = plt.Normalize(-10, 30)
    else:
        norm = plt.Normalize(-20, 30)"""
    norm = plt.Normalize(-5, 30)
    
    print(np.max(spec), np.min(spec), np.average(spec), np.median(spec))

    plt.axis('off')

    plt.imshow(spec[::-1], cmap=cmap, norm=norm, aspect='auto')
    #plt.imshow(spec[::-1], cmap=cmap, aspect='auto')
    #plt.imshow(spec[::-1], norm=norm, aspect='auto')
    #plt.imshow(spec[::-1], aspect='auto')

    #plt.colorbar(format='%+2.0f dB')
    plt.savefig(filename, bbox_inches = 'tight', pad_inches=0)


def _fft(data):
    data = np.ravel(data)
    return np.abs(np.fft.fft(data))

def wave_maxes(fftlist):#找峰值
    win = 5
    compare = 10

    amps = []
    for i in range(len(fftlist)):
        ampindex = np.argmax(fftlist[i:i+win])+i
        a = fftlist[ampindex]
        left = fftlist[ampindex-compare:ampindex]
        right = fftlist[ampindex+1:ampindex+compare+2]
        if left<[a] and right<[a]:
            if ampindex not in amps:
                amps.append(ampindex)
        i+=win
    return amps

class nine_one_weight_fft():#平滑fft變動幅度
    def __init__(self):
        self.i = True
        self.history = .0
    def spec(self, new):
        if self.i == True:
            self.history = copy.copy(new)
            self.i = False
        else:
            self.history = [a*0.7+b*0.3 for a, b in zip(self.history, new)]
        return self.history

def _median(sound):#找中位數
    weight_spec_mid = nine_one_weight_fft()
    sound_median = []

    for i in range(len(sound)):

        fftdata1 = _fft(sound[i])
        fftdata1 = weight_spec_mid.spec(fftdata1)
        if i%10 == 0:
            sound_median.append(fftdata1)

    return np.median(sound_median)

def get_mfcc_frame(sound, framestart, checkstart, absfile):
    frames = sound[framestart:checkstart]

    frames = np.array(frames)
    frames = np.ravel(frames)
    fmax = np.max(abs(frames))
    frames = frames/np.max(abs(frames))
    frames = frames*fmax

    spec = librosa.feature.mfcc(y=frames, sr=Fs,n_mfcc=40)
    
    dire = absfile[:absfile.rfind("/")]
    filename = absfile[absfile.rfind("/")+1:]
    
    
    if not os.path.exists(dire+"_mfcc"):
        os.mkdir(dire+"_mfcc")

    
    for j in range(10000):
        if not os.path.exists(dire+"_mfcc/"+filename+"_"+str(j)+".png"):
            plot_spec(spec, dire+"_mfcc/"+filename+"_"+str(j)+".png")
            break
    print("mfcc saved!!!")

def record(sound, mid, absfile):
    weight_spec = nine_one_weight_fft()
    voice = False#是否開始錄製
    endstart = True#聲音結束之後的時間點
    print(absfile)

    for i in range(len(sound)):
        #data = stream.read(CHUNK)
        #dataInt = struct.unpack(str(CHUNK) + 'h', data)
        #fft_spec
        fftdata1 = _fft(sound[i])
        fftdata1 = weight_spec.spec(fftdata1)
        #print(np.max(fftdata2))
        
        if i == len(sound)-1 and voice==True:
            voice = False
            if i-framestart >= Fs*5:#聲音如果太短就不要做圖
                get_mfcc_frame(sound, framestart, checkstart, absfile)
            else:
                print("at the last, too short!!")

        if np.max(fftdata1) > mid*6 and voice is False:

            print("start record!!!!")
            framestart = i
            voice = True

        elif voice is True:

            if i-framestart > Fs*6:#圖片太長就切段
                voice = False
                get_mfcc_frame(sound, framestart, checkstart, absfile)
                print("cut!!")

            elif np.max(fftdata1) < mid*6:
                if endstart is True:
                    checkstart = i#開始計算0.6秒
                    endstart = False

                checkend = i#0.6秒的最後
                cunt = checkend-checkstart

                if cunt > Fs*0.6:#原本23552為1.2秒
                    voice = False
                    if i-framestart >= Fs*3:#聲音如果太短就不要做圖
                        get_mfcc_frame(sound, framestart, framestart+Fs*6, absfile)
                
            else:
                endstart = True


if __name__ == '__main__':
    y, sr = librosa.load(r'C:\mfcc\iot\use\Oceanites520667.wav')
    print(len(y))
    mid = _median(y)
    print(mid)
    record(y, mid)

        
