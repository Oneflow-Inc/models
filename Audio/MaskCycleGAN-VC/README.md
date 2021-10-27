# MaskCycleGAN-VC with oneflow

Implementation of [MaskCycleGAN-VC](https://arxiv.org/pdf/2102.12841.pdf) with Oneflow.

Non-parallel voice conversion (VC) is a technique for training voice converters without a parallel corpus. MaskCycleGAN-VC is the state of the art method for non-parallel voice conversion using CycleGAN. It is trained using a novel auxiliary task of filling in frames (FIF) by applying a temporal mask to the input Mel-spectrogram. It demonstrates marked improvements over prior models such as CycleGAN-VC (2018), CycleGAN-VC2 (2019), and CycleGAN-VC3 (2020).


## Requirement

```bash
pip install -r requirements.txt
```


## Train

```bash
sh train.sh
```


## Infer

```bash
sh infer.sh
```
