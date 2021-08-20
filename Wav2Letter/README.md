# Wav2Letter Speech Recognition with oneflow

A oneflow implementation of Wav2Letter, a speech recognition model from Facebooks AI Research (FAIR) [paper](https://arxiv.org/pdf/1609.03193.pdf). 


## Requirements

```bash
pip install -r requirements.txt
```

## Data

We train and evaluate our models on [Google Speech Command Dataset](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data). 
This is a simple to use lightweight dataset for testing model performance.

### Instructions to download data

1. Download the dataset.
2. Create a `./speech_data` directory at root of this project.
3. Unzip the google speech data. Should be named `speech_commands_v0.01`. And move it to `./speech_data`.

### Data Preprocess

`data.py` contains scripts to process google speech command audio data into features compatible with Wav2Letter.

This will process the google speech commands audio data into 13 mfcc features with a max framelength of 250 (these are short audio clips). Anything less will be padded with zeros. Target data will be integer encoded and also padded to have the same length. Final outputs are numpy arrays saved as `x.npy` and `y.npy` in the `./speech_data` directory.


## Train

```bash
bash train.sh
```

## Infer

```bash
bash infer.sh
```

## Wer

oneflow 0.3031
pytorch 0.3099
