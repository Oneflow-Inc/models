"""
@author: Chenhao Lu <luchenhao@zhejianglab.com>
@author: Yizhang Wang <1739601638@qq.com>
"""
import argparse
import oneflow as flow

from model.deep_q_network import DeepQNetwork
from game.wrapped_flappy_bird import GameState
from model.utils import pre_processing
from random import random, randint, sample
import numpy


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird"""
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=84,
        help="The common width and height for all images",
    )
    parser.add_argument("--saved_path", type=str, default="checkpoints")

    args = parser.parse_args()
    return args


def test(opt):
    
    

    model = DeepQNetwork()
    pretrain_models = flow.load("{}".format(opt.saved_path))
    model.load_state_dict(pretrain_models)
    model.eval()
    model.to("cuda")
    game_state = GameState()
    image, reward, terminal = game_state.frame_step(0)
    image = pre_processing(
        image[: game_state.SCREENWIDTH, : int(game_state.BASEY)],
        opt.image_size,
        opt.image_size,
    )
    image = flow.Tensor(image)
    image = image.to("cuda")
    state = flow.cat(tuple(image for _ in range(4))).unsqueeze(0)

    while True:
        prediction = model(state)[0]
        action = flow.argmax(prediction).numpy()[0]

        next_image, reward, terminal = game_state.frame_step(action)
        next_image = pre_processing(
            next_image[: game_state.SCREENWIDTH, : int(game_state.BASEY)],
            opt.image_size,
            opt.image_size,
        )
        next_image = flow.Tensor(next_image)
        next_image = next_image.to("cuda")
        next_state = flow.cat((state[0, 1:, :, :], next_image)).unsqueeze(0)

        state = next_state


if __name__ == "__main__":
    opt = get_args()
    test(opt)
