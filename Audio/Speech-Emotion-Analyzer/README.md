# Speech-Emotion-Analyzer

Speech emotion recognition using LSTM and CNN, implemented in Oneflow.

Our code is inspired by the Keras implementation [Speech-Emotion-Recognition](https://github.com/Renovamen/Speech-Emotion-Recognition).


## Environments

- Python 3.6.11
- Oneflow 0.5.0


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
│   ├── opts.py              // argparse
│   └── plot.py              // plot graphs
├── config/                  // configure parameters
|   |——lstm.yaml             // configure parameters for lstm_ser model
│   └──cnn1d.yaml            // configure parameters for cnn1d_ser model
├── preprocess.py            // data preprocessing (extract features and store them locally)
├── train.py                 // train
├── train.sh                 // shell script for trainning
├── predict.py               // recognize the emotion of a given audio
└── predict.sh               // shell script for prediction

```

## Requirments

### Python
```python
pip install -r requirements.txt
```
### Tools

- [Opensmile](https://github.com/naxingyu/opensmile): extract features

## Data

We made use of CASIA dataset. However, the code can be easily adapted to any other dataset, including RAVEDSS, SAVEE, EMO-DB, etc.

For the access to the dataset used, please refer to [Issue about the dataset](https://github.com/Renovamen/Speech-Emotion-Recognition/issues/17) in the original Keras implementation.

## Usage

Once the dataset is downloaded, we can start training and prediction. 
### Train
```python
bash train.sh
```

### Predict
```python
bash predict.sh
```

## Performace
| Result | Onflow | Keras |
| --------- | ------- | ------- |
| Acc   | 82%+ |  80% |

## More Details

### Configuration

Parameters can be configured in the config files (YAML) under [`configs/`](https://github.com/Oneflow-Inc/models/tree/dev_audio_speech_emotion_analyzer/Audio/Speech-Emotion-Analyzer/configs).

It should be noted that, currently only the following 6 Opensmile standard feature sets are supported:

- `IS09_emotion`: [The INTERSPEECH 2009 Emotion Challenge](http://mediatum.ub.tum.de/doc/980035/292947.pdf), 384 features;
- `IS10_paraling`: [The INTERSPEECH 2010 Paralinguistic Challenge](https://sail.usc.edu/publications/files/schuller2010_interspeech.pdf), 1582 features;
- `IS11_speaker_state`: [The INTERSPEECH 2011 Speaker State Challenge](https://www.phonetik.uni-muenchen.de/forschung/publikationen/Schuller-IS2011.pdf), 4368 features;
- `IS12_speaker_trait`: [The INTERSPEECH 2012 Speaker Trait Challenge](http://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2012/Schuller12-TI2.pdf), 6125 features;
- `IS13_ComParE`: [The INTERSPEECH 2013 ComParE Challenge](http://www.dcs.gla.ac.uk/~vincia/papers/compare.pdf), 6373 features;
- `ComParE_2016`: [The INTERSPEECH 2016 Computational Paralinguistics Challenge](http://www.tangsoo.de/documents/Publications/Schuller16-TI2.pdf), 6373 features.

You may should modify item `FEATURE_NUM` in [`extract_feats/opensmile.py`](extract_feats/opensmile.py) if you want to use other feature sets.



### Preprocess

First of all, you should extract features of each audio in dataset and store them locally. Features extracted by Opensmile will be saved in `.csv` files and by librosa will be saved in `.p` files.

```python
python preprocess.py --config configs/example.yaml
```

where `configs/example.yaml` is the path to your config file



### Train

The path of the datasets can be configured in the config files (YAML) under [`configs/`](https://github.com/Oneflow-Inc/models/tree/dev_audio_speech_emotion_analyzer/Audio/Speech-Emotion-Analyzer/configs). Then:

```python
python train.py --config configs/example.yaml
```


### Predict

This is for when you have trained a model and want to predict the emotion for an audio, whose path can be configured in in the config files (YAML) under [`configs/`](https://github.com/Oneflow-Inc/models/tree/dev_audio_speech_emotion_analyzer/Audio/Speech-Emotion-Analyzer/configs)

```python
python predict.py --config configs/example.yaml
```


### Functions

#### Radar Chart

Plot a radar chart for demonstrating predicted probabilities.

```python
import utils
"""
Args:
    data_prob (np.ndarray): probabilities
    class_labels (list): labels
"""
utils.radar(data_prob, class_labels)
```


#### Play Audio

```python
import utils
utils.play_audio(file_path)
```


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


#### Waveform

Plot a waveform for an audio file.

```python
import utils
utils.waveform(file_path)
```


#### Spectrogram

Plot a spectrogram for an audio file.

```python
import utils
utils.spectrogram(file_path)
```
