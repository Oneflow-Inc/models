import os
import re
import sys
import librosa
import librosa.display
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    # fmin and fmax correspond to the minimum and the maximum basic frequency of human speech
    pitches, magnitudes = librosa.piptrack(X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # Spectral centroids
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # Spectral plane
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # The MFCC feature with coefficient being 50
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # Chromatography
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # Mel frequency
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)

    # ottava contrast
    contrast = np.mean(
        librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0
    )

    # zero-crossing rate
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # RMS energy
    rmse = librosa.feature.rmse(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array(
        [
            flatness,
            zerocr,
            meanMagnitude,
            maxMagnitude,
            meancent,
            stdcent,
            maxcent,
            stdMagnitude,
            pitchmean,
            pitchmax,
            pitchstd,
            pitch_tuning_offset,
            meanrms,
            maxrms,
            stdrms,
        ]
    )

    ext_features = np.concatenate(
        (ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast)
    )

    return ext_features


def extract_features(file: str, pad: bool = False) -> np.ndarray:
    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), "constant")
    return features(X, sample_rate)


def get_max_min(files: list) -> Tuple[float]:
    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_


def get_data_path(data_path: str, class_labels: list) -> list:
    """
    get path of all audio files

    Args:
        data_path (str): dataset folder path
        class_labels (list): emotion labels
    Returns:
        wav_file_path (list): path of all audio files
    """
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write("Curdir: %s\n" % cur_dir)
    os.chdir(data_path)

    # traverse the folder
    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        # read audio files in this folder
        for filename in os.listdir("."):
            if not filename.endswith("wav"):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir("..")
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path


def load_feature(
    config, feature_path: str, train: bool
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    load feature data for `csv` file

    Args:
        config: configuration items
        feature_path (str): path of feature data
        train (bool): training data

    Returns:
        - X (Tuple[np.ndarray]): training feature, testing fearture and labels
        - X (np.ndarray): predicting feature
    """
    features = pd.DataFrame(
        data=joblib.load(feature_path), columns=["file_name", "features", "emotion"]
    )

    X = list(features["features"])
    Y = list(features["emotion"])

    # standardize the path of the model
    # if os.path.isdir(config.checkpoint_path):
    # os.makedirs(config.checkpoint_path)
    scaler_path = os.path.join(config.checkpoint_path, "SCALER_Librosa.m")

    if train == True:
        # standardize the data
        scaler = StandardScaler().fit(X)
        # save
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return x_train, x_test, y_train, y_test

    else:

        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X


def get_data(
    config, data_path: str, feature_path: str, train: bool
) -> Union[Tuple[np.ndarray], np.ndarray]:
    """
    Extract all the audio features: Go through all the folders, read the audio in each folder,
    extract the features of each audio, and save all the features to the path: `feature_path`
    Args:
        confi: configuration items
        data_path (str): Dataset folder path
        feature_path (str): path of feature data
        train (bool): training data

    Returns:
        - train = True: training feature, testing fearture and labels
        - train = False: predicting feature
    """
    if train == True:
        files = get_data_path(data_path, config.class_labels)
        max_, min_ = get_max_min(files)

        mfcc_data = []
        for file in files:
            label = re.findall(".*-(.*)-.*", file)[0]

            features = extract_features(file, max_)
            mfcc_data.append([file, features, config.class_labels.index(label)])

    else:
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    cols = ["file_name", "features", "emotion"]
    mfcc_pd = pd.DataFrame(data=mfcc_data, columns=cols)
    pickle.dump(mfcc_data, open(feature_path, "wb"))

    return load_feature(config, feature_path, train=train)
