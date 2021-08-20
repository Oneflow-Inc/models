import os
import random
import pickle

import numpy as np
from sonopy import mfcc_spec
from scipy.io.wavfile import read
from tqdm import tqdm


class IntegerEncode:
    """Encodes labels into integers
    
    Args:
        labels (list): shape (n_samples, strings)
    """

    def __init__(self, labels):
        # reserve 0 for blank label
        self.char2index = {
            "-": 0,
            "pad":1
        }
        self.index2char = {
            0: "-",
            1: "pad"
        }
        self.grapheme_count = 2
        self.process(labels)
        self.max_label_seq = 6

    def process(self, labels):
        """builds the encoding values for labels
        
        Args:
            labels (list): shape (n_samples, strings)
        """
        strings = "".join(labels)
        for s in strings:
            if s not in self.char2index:
                self.char2index[s] = self.grapheme_count
                self.index2char[self.grapheme_count] = s
                self.grapheme_count += 1

    def convert_to_ints(self, label):
        """Convert into integers
        
        Args:
            label (str): string to encode
        
        Returns:
            list: shape (max_label_seq)
        """
        y = []
        for char in label:
            y.append(self.char2index[char])
        if len(y) < self.max_label_seq:
            diff = self.max_label_seq - len(y)
            pads = [self.char2index["pad"]] * diff
            y += pads
        return y

    def save(self, file_path):
        """Save integer encoder model as a pickle file

        Args:
            file_path (str): path to save pickle object
        """
        file_name = os.path.join(file_path, "int_encoder.pkl")
        with open(file_name, 'wb') as f:
            pickle.dump(self.__dict__, f)


def normalize(values):
    """Normalize values to mean 0 and std 1
    
    Args:
        values (np.array): shape (frame_len, features)
    
    Returns:
        np.array: normalized features
    """
    return (values - np.mean(values)) / np.std(values)


class GoogleSpeechCommand():
    """Data set can be found here 
        https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data
    """

    def __init__(self, data_path="speech_data/speech_commands_v0.01", sr=16000):
        self.data_path = data_path
        self.labels = [
            'right', 'eight', 'cat', 'tree', 'bed', 'happy', 'go', 'dog', 'no', 
            'wow', 'nine', 'left', 'stop', 'three', 'sheila', 'one', 'bird', 'zero',
            'seven', 'up', 'marvin', 'two', 'house', 'down', 'six', 'yes', 'on', 
            'five', 'off', 'four'
        ]
        self.intencode = IntegerEncode(self.labels)
        self.sr = sr
        self.max_frame_len = 225

    def get_data(self, progress_bar=True):
        """Currently returns mfccs and integer encoded data

        Returns:
            (list, list): 
                inputs shape (sample_size, frame_len, mfcs_features)
                targets shape (sample_size, seq_len)  seq_len is variable
        """
        pg = tqdm if progress_bar else lambda x: x

        inputs, targets = [], []
        meta_data = []
        for labels in self.labels:
            path = os.listdir(os.path.join(self.data_path, labels))
            for audio in path:
                audio_path = os.path.join(self.data_path, labels, audio)
                meta_data.append((audio_path, labels))
        
        random.shuffle(meta_data)

        for md in pg(meta_data):
            audio_path = md[0]
            labels = md[1]
            _, audio = read(audio_path)
            mfccs = mfcc_spec(
                audio, self.sr, window_stride=(160, 80),
                fft_size=512, num_filt=20, num_coeffs=13
            )
            mfccs = normalize(mfccs)
            diff = self.max_frame_len - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")
            inputs.append(mfccs)

            target = self.intencode.convert_to_ints(labels)
            targets.append(target)
        return inputs, targets

    @staticmethod
    def save_vectors(file_path, x, y):
        """saves input and targets vectors as x.npy and y.npy
        
        Args:
            file_path (str): path to save numpy array
            x (list): inputs
            y (list): targets
        """
        x_file = os.path.join(file_path, "x")
        y_file = os.path.join(file_path, "y")
        np.save(x_file, np.asarray(x))
        np.save(y_file, np.asarray(y))

    @staticmethod
    def load_vectors(file_path):
        """load inputs and targets
        
        Args:
            file_path (str): path to load targets from
        
        Returns:
            inputs, targets: np.array, np.array
        """
        x_file = os.path.join(file_path, "x.npy")
        y_file = os.path.join(file_path, "y.npy")

        inputs = np.load(x_file)
        targets = np.load(y_file)
        return inputs, targets


if __name__ == "__main__":
    gs = GoogleSpeechCommand()
    inputs, targets = gs.get_data()
    gs.save_vectors("./speech_data", inputs, targets)
    gs.intencode.save("./speech_data")
    print("preprocessed and saved")
