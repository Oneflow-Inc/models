import os
import random

import librosa
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import oneflow as flow
from oneflow.utils.data.dataloader import DataLoader
from oneflow.utils.data.dataset import Dataset

from utils.preprocess import FEATURE_DIM, FFTSIZE, FRAMES, SAMPLE_RATE, world_features
from utils.utility import Normalizer, speakers


class AudioDataset(Dataset):
    """docstring for AudioDataset."""

    def __init__(self, datadir: str):
        super(AudioDataset, self).__init__()
        self.datadir = datadir
        self.files = librosa.util.find_files(datadir, ext='npy')
        self.encoder = LabelBinarizer().fit(speakers)

    def __getitem__(self, idx):
        p = self.files[idx]
        filename = os.path.basename(p)
        speaker = filename.split(sep='_', maxsplit=1)[0]
        label = self.encoder.transform([speaker])[0]
        mcep = np.load(p)

        mcep = flow.Tensor(mcep)
        mcep = flow.unsqueeze(mcep, 0)

        return mcep, flow.tensor(speakers.index(speaker), dtype=flow.long), flow.tensor(label, dtype=flow.float)

    def speaker_encoder(self):
        return self.encoder

    def __len__(self):
        return len(self.files)


def data_loader(datadir: str, batch_size=4, shuffle=True, mode='train', num_workers=0):
    '''if mode is train datadir should contains training set which are all npy files
        or, mode is test and datadir should contains only wav files.
    '''
    dataset = AudioDataset(datadir)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)

    return loader


class TestSet(object):
    """docstring for TestSet."""

    def __init__(self, datadir: str):
        super(TestSet, self).__init__()
        self.datadir = datadir
        self.norm = Normalizer()

    def choose(self):
        '''choose one speaker for test'''
        r = random.choice(speakers)
        return r

    def test_data(self, src_speaker=None):
        '''choose one speaker for conversion'''
        if src_speaker:
            r_s = src_speaker
        else:
            r_s = self.choose()
        p = os.path.join(self.datadir, r_s)
        wavfiles = librosa.util.find_files(p, ext='wav')

        res = {}
        for f in wavfiles:
            filename = os.path.basename(f)
            wav, _ = librosa.load(f, sr=SAMPLE_RATE, dtype=np.float64)
            f0, timeaxis, sp, ap, coded_sp = world_features(
                wav, SAMPLE_RATE, FFTSIZE, FEATURE_DIM)
            coded_sp_norm = self.norm.forward_process(coded_sp.T, r_s)

            if not res.__contains__(filename):
                res[filename] = {}
            res[filename]['coded_sp_norm'] = np.asarray(coded_sp_norm)
            res[filename]['f0'] = np.asarray(f0)
            res[filename]['ap'] = np.asarray(ap)
        return res, r_s
