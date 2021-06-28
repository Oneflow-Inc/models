# Playing Flappy Bird Using Deep Reinforcement Learning

Implementation of DQN to play the Flappy Bird game with [Oneflow](https://github.com/Oneflow-inc/oneflow/) framework .

This work is based on: [uvipen/Flappy-bird-deep-Q-learning-pytorch](https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch) and [Ldpe2G/DeepLearningForFun](https://github.com/Ldpe2G/DeepLearningForFun/tree/bb532da54df8d65471099222c0b534b2210315d5/Oneflow-Python/DRL-FlappyBird)

## screenshots
<img src="https://github.com/Oneflow-Inc/models/blob/6a8610162fc916ad36c8ecc73470189f2beea49f/FlappyBird/demo/play.gif"/>


## Environment
| Spec                        |                                                             |
|-----------------------------|-------------------------------------------------------------|
| Operating System            | Ubuntu 18.04                                        |
| GPU                         | Nvidia A100-SXM4-40GB                          |
| CUDA Version                | 11.2                                                   |
| Driver Version              | 460.73.01                                             |
| Oneflow Version 	          | branch: master, commit_id: 90d3277a098f483d0a0e68621b7c8fb2497a6fc2 |

## Requirements

* python3
    - pygame
    - numpy
    - opencv
* [Oneflow](https://github.com/Oneflow-inc/oneflow)

## Pretrain Model

[DQN_FlappyBird.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/model_zoo/rl/flappybird_pretrain_model.zip)

## Play with pretrain model

```bash
bash test_flappy_bird.sh
```

## Train from scratch

```bash
bash train_flappy_bird.sh
```

It will take about at least 40000 time steps before the bird can learn to play the game, be patients :).




