# speaker recognization demo
This is a demo of the speaker recognition question in the ZhejiangLab AI Voice Contest implementated with oneflow.


## Datasets:

The demo data used here is the case where only two speakers are included.
`data_preprocess.py` contains scripts to process the demo data into features compatible with Wav2Letter.
Data preprocessing includes segmenting the voice according to the time period divided in the txt file, and generating the corresponding relationship between the voice segment and the speaker.

## Data preprocessed:

```bash
sh data_preprocess.sh
```

## Start training process:

```bash
sh train.sh
```

## Model inference, this one is configured to process `data/test_data/name1_15.wav` file.

```bash
sh infer.sh
```
