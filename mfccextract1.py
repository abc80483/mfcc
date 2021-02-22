import librosa
import argparse
from librosa import display, core
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
import os
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

#一次移動距離
move = 20000
#片段長度
segmentation = 50000

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in","--input-folder", dest="input_folder", required=False,
                        help="Input folder directory", default=".")
    parser.add_argument("-ex","--extract", dest="extract", required=True,
                        help="choose mfcc or fft")
    return parser

def get_label(input_folder):
    #將聲音檔案的資料夾當成類別的標籤回傳
    if not input_folder.endswith("/"):
        input_folder = input_folder+"/"

    labels = []

    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue
        label = subfolder[subfolder.rfind("/")+1:]
        labels.append(label)
    return labels

def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

def filename(dir):
    #使用一個list來存所有wav檔的檔名
    files = []
    for root, _, file in os.walk(dir):
        for f in file:
            absfile = os.path.join(root, f)
            if absfile.endswith(".wav"):
                absfile = absfile.replace("\\", "/")
                files.append(absfile)
    return files

class mfccextract:
    def __init__(self, move, segmentation):
        self.move = move
        self.segmentation = segmentation
        parser = build_arg_parser()
        args = parser.parse_args()
        input_folder = args.input_folder

        if args.extract == "mfcc":
            self.ex = "mfcc"
        elif args.extract == "fft":
            self.ex = "fft"
        else:
            print("-ex or --extract only support fft and mfcc!")
            exit()

        self.labels = get_label(input_folder)
        self.files = filename(input_folder)
        print(self.files)

    def extract(self):
        path = "dataset"

        if not os.path.exists(path):
            os.mkdir(path)
        for absfile in self.files:
            bird = ""
            for label in self.labels:
                if absfile.rfind(label)!=-1:
                    bird = label
            y, sr = librosa.load(absfile)

            y = normalize_audio(y)

            for i in range(1, len(y), self.move):
                if(i+self.segmentation<y.shape[0]):
                    newy = y[i:i + self.segmentation]

                    if self.ex == "mfcc":
                        spec = librosa.feature.mfcc(newy, n_mels=40, fmax=80000)
                    elif self.ex == "fft":
                        spec = np.abs(core.stft(newy))

                    newspec = sklearn.preprocessing.scale(spec, axis=1)
                    plt.figure(figsize=(5, 4))
                    plt.clf()
                    librosa.display.specshow(newspec)
                    plt.axis("off")
                    #plt.plot(np.linspace(0, len(newy) / sr, num=len(newy)), newy)

                    plt.savefig(path+"/"+bird+"_"+str(i)+".png")
                    print("\""+bird+"_"+str(i)+".png\""+" has been saved!")



if __name__ == '__main__':
    ok = mfccextract(move, segmentation)
    ok.extract()
