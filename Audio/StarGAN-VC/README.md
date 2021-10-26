# StarGAN-VC with oneflow

Implementation of [StarGAN-VC](https://arxiv.org/abs/1806.02169) with Oneflow.

StarGAN-VC is noteworthy in that it (1) requires no parallel utterances, transcriptions, or time alignment procedures for speech generator training, (2) simultaneously learns many-to-many mappings across different attribute domains using a single generator network, (3) is able to generate converted speech signals quickly enough to allow real-time implementations and (4) requires only several minutes of training examples to generate reasonably realistic-sounding speech. 


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
