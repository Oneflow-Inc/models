# Speech-Emotion-Analyzer

Speech emotion recognition using LSTM and CNN, implemented in Oneflow.

Our code is inspired by the Keras implementation [Speech-Emotion-Recognition](https://github.com/Renovamen/Speech-Emotion-Recognition).

&nbsp;

## Environments

- Python 3.6.11
- Oneflow 0.5.0

&nbsp;

## Structure

```
├── models
│   ├── lstm_ser.py          // Lstm model used for speech emotion recongnition
│   ├── lstm_oneflow.py      // Lstm operator
│   └── cnn1d_ser.py         // Cnn1D model used for speech emotion recongnition
├── extract_feats/           // features extraction
│   ├── librosa.py           // extract features using librosa
│   └── opensmile.py         // extract features using Opensmile
├── utils/
│   ├── files.py             // setup dataset (classify and rename)
│   ├── opts.py              // argparse
│   └── plot.py              // plot graphs
├── config/                  // configure parameters
|   |——lstm.yaml             // configure parameters for lstm_ser model
│   └──cnn1d.yaml            // configure parameters for cnn1d_ser model
├── preprocess.py            // data preprocessing (extract features and store them locally)
├── train_oneflow.py         // train
└── predict.py               // recognize the emotion of a given audio

```

&nbsp;

## Requirments

### Python
- [oneflow](https://github.com/Oneflow-Inc): Lstm and Cnn
- [librosa](https://github.com/librosa/librosa): extract features, waveform
- [SciPy](https://github.com/scipy/scipy): spectrogram
- [pandas](https://github.com/pandas-dev/pandas): Load features
- [Matplotlib](https://github.com/matplotlib/matplotlib): plot graphs
- [numpy](github.com/numpy/numpy)

### Tools

- [Opensmile](https://github.com/naxingyu/opensmile): extract features

&nbsp;

## Data

We made use of CASIA dataset. However, the code can be easily adapted to any other dataset, including RAVEDSS, SAVEE, EMO-DB, etc.
&nbsp;

## Usage

### Prepare

Install dependencies:

```python
pip install -r requirements.txt
```

Install [Opensmile](https://github.com/naxingyu/opensmile).

&nbsp;

### Configuration

Parameters can be configured in the config files (YAML) under [`configs/`](https://github.com/yingzhao27/Speech-Emotion-Analyzer-with-oneflow/tree/main/configs).

It should be noted that, currently only the following 6 Opensmile standard feature sets are supported:

- `IS09_emotion`: [The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), 384 features;
- `IS10_paraling`: [The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf), 1582 features;
- `IS11_speaker_state`: [The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf), 4368 features;
- `IS12_speaker_trait`: [The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf), 6125 features;
- `IS13_ComParE`: [The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf), 6373 features;
- `ComParE_2016`: [The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), 6373 features.

You may should modify item `FEATURE_NUM` in [`extract_feats/opensmile.py`](extract_feats/opensmile.py) if you want to use other feature sets.

&nbsp;

### Preprocess

First of all, you should extract features of each audio in dataset and store them locally. Features extracted by Opensmile will be saved in `.csv` files and by librosa will be saved in `.p` files.

```python
python preprocess.py --config configs/example.yaml
```

where `configs/example.yaml` is the path to your config file

&nbsp;

### Train

The path of the datasets can be configured in [`config.py`](config.py). Audios which express the same emotion should be put in the same folder (you may want to refer to [`utils/files.py`](utils/files.py) when setting up datasets), for example:

```
└── datasets
    ├── angry
    ├── happy
    ├── sad
    ...
```

Then:

```python
python train.py --config configs/example.yaml
```

&nbsp;

### Predict

This is for when you have trained a model and want to predict the emotion for an audio.

First modify following things in [`predict.py`](predict.py):

```python
audio_path = 'str: path_to_your_audio'
```

Then:

```python
python predict.py --config configs/example.yaml
```

&nbsp;

### Functions

#### Radar Chart

Plot a radar chart for demonstrating predicted probabilities.

Source: [Radar](https://github.com/Zhaofan-Su/SpeechEmotionRecognition/blob/master/leidatu.py)

```python
import utils
"""
Args:
    data_prob (np.ndarray): probabilities
    class_labels (list): labels
"""
utils.radar(data_prob, class_labels)
```

&nbsp;

#### Play Audio

```python
import utils
utils.play_audio(file_path)
```

&nbsp;

#### Plot Curve

Plot loss curve or accuracy curve.

```python
import utils
"""
Args:
    train (list): loss or accuracy on train set
    val (list): loss or accuracy on validation set
    title (str): title of figure
    y_label (str): label of y axis
"""
utils.curve(train, val, title, y_label)
```

&nbsp;

#### Waveform

Plot a waveform for an audio file.

```python
import utils
utils.waveform(file_path)
```

&nbsp;

#### Spectrogram

Plot a spectrogram for an audio file.

```python
import utils
utils.spectrogram(file_path)
```

&nbsp;

### Performace
We achieved recognition accuracy above 80%, which is in accordance with 80% accuracy claimed by the original Keras implementation.
