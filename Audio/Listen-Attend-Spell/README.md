# Listen, Attend and Spell
An Onflow implementation of Listen, Attend and Spell (LAS) [1], an end-to-end automatic speech recognition framework, which directly converts acoustic features to character sequence using only one nueral network.

Our code is inspired by the Pytorch implementation [Listen-Attend-Spell](https://github.com/kaituoxu/Listen-Attend-Spell).

## Install
- Python3 (Recommend Anaconda)
- Oneflow 0.5.0
- [Kaldi](https://github.com/kaldi-asr/kaldi) (Just for feature extraction)
- `cd tools; make KALDI=/path/to/kaldi`
- If you want to run `egs/aishell/run.sh`, download [aishell](http://www.openslr.org/33/) dataset for free.

## Usage
### Quick start
1. `cd egs/aishell` and modify aishell data path to your path in `run.sh`.
2. `bash run.sh`, that's all!

You can change hyper-parameter by `bash run.sh --parameter_name parameter_value`. For example, `bash run.sh --stage 3`. See parameter name in `egs/aishell/run.sh` before `. utils/parse_options.sh`.

### Workflow
Workflow of `egs/aishell/run.sh`:
- Stage 0: Data Preparation
- Stage 1: Feature Generation
- Stage 2: Dictionary and Json Data Preparation
- Stage 3: Network Training
- Stage 4: Decoding

### More detail
`egs/aishell/run.sh` provide example usage.
```bash
cd egs/aishell/
. ./path.sh
```
Train
```bash
train.py -h
```
Decode
```bash
recognize.py -h
```

### How to visualize loss?
If you want to visualize your loss, you can make use of [loss_visualize.py](egs/aishell/loss_visualize.py), in which you can change parameters by `python loss_visualize.py --parameter_name parameter_value`.

## Reference
[1] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in ICASSP 2016. (https://arxiv.org/abs/1508.01211v2)
