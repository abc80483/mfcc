import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import args_practice
import warnings

warnings.filterwarnings("ignore", message="Numerical issues were encountered ")

#一次移動距離
move = 20000
#片段長度
segmentation = 50000

class mfccextract:
    def __init__(self, move, segmentation):
        self.move = move
        self.segmentation = segmentation
        parser = args_practice.build_arg_parser()
        args = parser.parse_args()
        input_folder = args.input_folder

        if args.extract == "mfcc":
            pass
        elif args.extract == "fft":
            pass
        else:
            print("-ex or --extract only support fft and mfcc!")
            exit()
        labels = args_practice.get_label(input_folder)
        self.files = args_practice.filename(self.move, self.segmentation, input_folder)
        print(self.files)

        self.extract()

    def extract(self):
        for absfile in self.files:
            y, sr = librosa.load(absfile)
            y = args_practice.normalize_audio(y)

            for i in range(1, len(y), self.move):
                if(i+self.segmentation<y.shape[0]):
                    newy = y[i:i + self.segmentation]
                    mfcc = librosa.feature.mfcc(newy, n_mels=40, fmax=80000)
                    newmfcc = sklearn.preprocessing.scale(mfcc, axis=1)
                    plt.figure(figsize=(5, 4))
                    plt.clf()
                    librosa.display.specshow(newmfcc)
                    plt.axis("off")
                    #plt.plot(np.linspace(0, len(newy) / sr, num=len(newy)), newy)
                    plt.savefig('amptitude'+str(i)+".png")
                    print("\"amptitude"+str(i)+".png\""+" has been save!")



if __name__ == '__main__':
    ok = mfccextract(move, segmentation)
