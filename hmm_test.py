import os
import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from hmmlearn import hmm


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trainning')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
                        help="Input folder containing the audio files in subfolders")
    return parser

class HMMTrainer(object):
    def __init__(self, model_name="GaussianHMM", n_components=4, conv_type="diag", n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.conv_type = conv_type
        self.n_iter = n_iter
        self.models = []
        
        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components,
                                         covariance_type=self.conv_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')
    
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))
        
    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    hmm_models = []

    #parse the input directory
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder):
            continue

        label = subfolder[subfolder.rfind('/') + 1:]
        
        X = np.array([])
        y_words = []
        warning.filterwarnings("ignore")
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            #read the input data
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            mfcc_features = mfcc(audio, sampling_freq)

            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            y_words.append(label)
        
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append(hmm_trainer)
        hmm_trainer = None
        
    input_files = [
        
        ]
    
    for input_file in input_files:
        sampling_freq, audio = wavfile.read(input_file)
        
        mfcc_features = mfcc(audio, sampling_freq)

        max_score = [float("-inf")]
        output_label = [float("-inf")]
        
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label

        print( "\nTrue", input_file[input_file,find('/')])