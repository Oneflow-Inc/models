# CycleGan-VC2 with oneflow

Implementation of [CycleGan-VC2](https://arxiv.org/abs/1904.04631) with Oneflow.

Non-parallel voice conversion (VC) is a technique for **learning the mapping from source to target speech without relying on parallel data**.
CycleGAN-VC2 is an improved version of CycleGAN-VC incorporating three new techniques: an improved objective (two-step adversarial losses), improved generator (2-1-2D CNN), and improved discriminator (PatchGAN).


## Requirement

```bash
pip install -r requirements.txt
```


## Train

```bash
bash train.sh
```


## Infer

```bash
bash infer.sh
```

