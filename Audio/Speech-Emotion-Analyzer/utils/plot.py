import wave
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.io.wavfile as wav
import numpy as np


def play_audio(file_path: str) -> None:
    """
    play the audio

    Args:
        file_path (str): audio file path
    """
    import pyaudio

    p = pyaudio.PyAudio()
    f = wave.open(file_path, "rb")
    stream = p.open(
        format=p.get_format_from_width(f.getsampwidth()),
        channels=f.getnchannels(),
        rate=f.getframerate(),
        output=True,
    )
    data = f.readframes(f.getparams()[3])
    stream.write(data)
    stream.stop_stream()
    stream.close()
    f.close()


def curve(train: list, val: list, title: str, y_label: str) -> None:
    """
    Draw loss and accuracy curve

    Args:
        train (list): loss/acc of training set
        val (list): loss/acc of testing set
        title (str): title of figure
        y_label (str): title of y axis
    """
    plt.plot(train)
    plt.plot(val)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def radar(data_prob: np.ndarray, class_labels: list) -> None:
    """
    Draw the radar chart of predicted probability

    Args:
        data_prob (np.ndarray): probability array
        class_labels (list): emotion labels
    """
    angles = np.linspace(0, 2 * np.pi, len(class_labels), endpoint=False)
    data = np.concatenate((data_prob, [data_prob[0]]))  # closure
    angles = np.concatenate((angles, [angles[0]]))  # closure
    class_labels = np.concatenate((class_labels, [class_labels[0]]))

    fig = plt.figure()

    # polar parameters
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, data, "bo-", linewidth=2)
    ax.fill(angles, data, facecolor="r", alpha=0.25)
    ax.set_thetagrids(angles * 180 / np.pi, class_labels)
    ax.set_title("Emotion Recognition", va="bottom")

    # set the maximum data value of the radar chart
    ax.set_rlim(0, 1)
    ax.grid(True)
    plt.show()


def waveform(file_path: str) -> None:
    """
    Draw audio waveform

    Args:
        file_path (str): audio file path
    """
    data, sampling_rate = librosa.load(file_path)
    plt.figure(figsize=(15, 5))
    display.waveplot(data, sr=sampling_rate)
    plt.show()


def spectrogram(file_path: str) -> None:
    """
    Dram spectrogram

    Args:
        file_path (str): audio file path
    """

    # sr: sampling rate
    # x: numpy array of audio data
    sr, x = wav.read(file_path)

    # step: 10ms, window: 30ms
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin
    window = np.hamming(nwin)

    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft // 2))

    for i, n in enumerate(nn):
        xseg = x[n - nwin : n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[: nfft // 2]))

    plt.imshow(X.T, interpolation="nearest", origin="lower", aspect="auto")
    plt.show()
