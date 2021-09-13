# AM-MobileNet1D with oneflow

Implementation of [AM-MobileNet1D](https://arxiv.org/abs/2004.00132) with Oneflow.

The mobile computation requires applications with reduced storage size, non-processing and memory intensive and efficient energy-consuming. The deep learning approaches, in contrast, usually are energy expensive, demanding storage, processing power, and memory. Additive Margin MobileNet1D (AM-MobileNet1D) is **a portable and effective model**. It obtains equivalent or better performances on TIMIT and MIT datasets concerning the baseline methods. Additionally, it takes only 11.6 megabytes on disk storage against 91.2 from SincNet and AM-SincNet architectures, making the model **seven times faster, with eight times fewer parameters**.


## Prerequisites

- Python 3.8
- oneflow 0.5.0
- pysoundfile


## Datasets

We use the TIMIT dataset to train the model. However, the code can be easily adapted to any speech dataset.

`utils/TIMIT_preparation.py` contains scripts to process TIMIT dataset into features compatible with AM-MobileNet1D.

This file receives three parameters, $TIMIT_FOLDER、$DATA_PREPARED and $LABEL. *$TIMIT_FOLDER* is the folder of the original TIMIT corpus. *$OUTPUT_FOLDER* is the folder in which the preprocessed TIMIT will be stored. *$LABEL* is the list of the TIMIT files used for training/test the speaker id system.


## Train

```bash
bash train.sh
```


## Infer

```bash
bash infer.sh
```


## Accracy

oneflow 0.7953
pytorch 0.7870

