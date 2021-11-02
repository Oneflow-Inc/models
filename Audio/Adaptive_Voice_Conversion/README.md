# One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization
This is an Oneflow implementation of the paper [One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization](https://arxiv.org/abs/1904.05742). By separately learning speaker and content representations, we can achieve one-shot VC by only one utterance from source speaker and one utterace from target speaker. You can download the pretrain model from [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/audio/Adaptive_Voice_Conversion.zip) and the coresponding normalization parameters for inference from [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/Audio/attr.zip).

Our code is inspired by the Pytorch implementation [adaptive_voice_conversion](https://github.com/jjery2243542/adaptive_voice_conversion).


# Dependencies
- python 3.6+
- oneflow 0.5.0
- numpy 1.16.0
- librosa 0.6.3
- SoundFile 0.10.2 
We also use some preprocess script from [Kyubyong/tacotron](https://github.com/Kyubyong/tacotron) and [magenta/magenta/models/gansynth](https://github.com/tensorflow/magenta/tree/master/magenta/models/gansynth).

# Data
We make use of VCTK dataset, which can be downloaded from  [CSTR VCTK Corpus](https://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html)

# Preprocess
```bash
bash preprocess/preprocess_vctk.sh
```
The preprocess code is at `preprocess/`.The configuation for preprocessing is at `preprocess/vctk.config`. 
where:
- **segment\_size** is the segment size for training. Default: 128
- **data\_dir** is the directory to put preprocessed files. 
- **raw\_data\_dir** is the directory to put the raw data. 
- **n_out_speakers** is the number of speakers for testing. Default: 20.
- **test\_prop** is the proportion for validation utterances. Default: 0.1
- **training\_samples** is the number of sampled segments for training (we sample it in the preprocess stage). Default: 10000000.
- **testing_samples** is the number of sampled segments for testing. Default: 10000.
- **n\_utt\_attr** is the number of utterances to compute mean and standard deviation for normalization. Default: 5000.
- **train_set**: only for LibriTTS. The subset used for training. Default: train-clean-100.
- **test_set**: only for LibriTTS. The subset used for testing. Default: dev-clean.

Once you edited the config file, you can run `preprocess_vctk.sh`  to preprocess the dataset. 
<br>
Also, you can change the feature extraction config in `preprocess/tacotron/hyperparams.py`

# Training
```bash
bash train.sh
```
The default arguments can be found in `train.sh`. The usage of each arguments are listed below. 
- **-c**: the path of config file, the default hyper-parameters can be found at `config.yaml`.
- **-iters**: train the model with how many iterations. default: 200000
- **-summary_steps**: record training loss every n steps.
- **-train_set**: the data file for training (`train` if the file is train.pkl). Default: `train`
- **-train_index_file**: the name of training index file. Default: `train_samples_128.json`
- **-data_dir**: the directory for processed data.
- **-store_model_path**: the path to store the model.

# Inference
```bash
bash infer.sh
```
- **-c**: the path of config file.
- **-m**: the path of model checkpoint.
- **-a**: the attribute file for normalization ad denormalization.
- **-s**: the path of source file (.wav).
- **-t**: the path of target file (.wav).
- **-o**: the path of output converted file (.wav).

# Reference
Ju-chieh Chou, Cheng-chieh Yeh, Hung-yi Lee. One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization[J]. arXiv preprint arXiv:1904.05742, 2019.
