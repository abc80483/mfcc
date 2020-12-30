import os
import argparse
import librosa
import numpy as np
from hmmlearn import hmm


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in","--input-folder", dest="input_folder", required=False,
                        help="Input folder directory", default=".")
    parser.add_argument("-ex","--extract", dest="extract", required=True,
                        help="choose mfcc or fft")
    return parser

def get_label(input_folder):
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

def filename(move, segmentation, dir):
    files = []
    for root, _, file in os.walk(dir):
        for f in file:
            absfile = os.path.join(root, f)
            if absfile.endswith(".wav"):
                absfile = absfile.replace("\\", "/")
                files.append(absfile)
    return files

class HMMTrainer:
    def __init__(self, model_name="GaussianHMM", n_components=4, conv_type="diag", n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.conv_type = conv_type
        self.n_iter = n_iter
        self.models = []

        if model_name=="GaussianHMM":
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.conv_type, n_iter=self.n_iter)
        else:
            raise TypeError("Invalid model type")

    def train(self, X):
        np.seterr(all='ignore')
        self.model.append(self.model.fit(X))


    def get_score(self, input_data):
        return self.model.score(input_data)
