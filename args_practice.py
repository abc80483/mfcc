import os
import argparse
import librosa
import numpy as np

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in","--input-folder", dest="input_folder", required=True,
                        help="Input folder directory")
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
