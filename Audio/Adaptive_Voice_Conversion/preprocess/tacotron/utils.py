# -*- coding: utf-8 -*-
# /usr/bin/python2
"""
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
"""
from __future__ import print_function, division
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from .hyperparams import Hyperparams as hp
import numpy as np
import librosa
import copy
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from scipy import signal
import os


def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def get_spectrograms(fpath):
    """Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    """

    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=hp.top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(
        y=y, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length
    )

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag


def melspectrogram2wav(mel):
    """# Generate wave file from spectrogram"""
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(hp.sr, hp.n_fft, hp.n_mels)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def spectrogram2wav(mag):
    """ Generate wave file from spectrogram"""
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    """Applies Griffin-Lim's raw.
    """
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    """
    spectrogram: [f, t]
    """
    return librosa.istft(
        spectrogram, hp.hop_length, win_length=hp.win_length, window="hann"
    )


def plot_alignment(alignment, gs):
    """Plots the alignment
    alignments: A list of (numpy) matrix of shape (encoder_steps, decoder_steps)
    gs : (int) global step
    """
    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im)
    plt.title("{} Steps".format(gs))
    plt.savefig("{}/alignment_{}k.png".format(hp.logdir, gs // 1000), format="png")


def load_spectrograms(fpath):
    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0  # for reduction
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")
    return fname, mel.reshape((-1, hp.n_mels * hp.r)), mag
