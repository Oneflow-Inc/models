"""
@author: Chenhao Lu <luchenhao@zhejianglab.com>
@author: Yizhang Wang <1739601638@qq.com>
"""
import argparse
import os
from random import random, randint, sample

import numpy as np
import oneflow as flow

from model.deep_q_network import DeepQNetwork
from game.wrapped_flappy_bird import GameState
from model.utils import pre_processing


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
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of images per batch"
    )
    parser.add_argument(
        "--optimizer", type=str, choices=["sgd", "adam"], default="adam"
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument(
        "--replay_memory_size",
        type=int,
        default=50000,
        help="Number of epoches between testing phases",
    )
    parser.add_argument("--save_checkpoint_path", type=str, default="checkpoints")

    args = parser.parse_args()
    return args


def train(opt):

    # Step 1: init BrainDQN
    model = DeepQNetwork()
    model.to("cuda")
    optimizer = flow.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = flow.nn.MSELoss()
    criterion.to("cuda")

    # Step 2: init Flappy Bird Game
    game_state = GameState()
    # Step 3: play game
    # image.shape = (288,512,3), reward: float, terminal: boolean
    image, reward, terminal = game_state.frame_step(0)
    # image.shape = (84, 84)
    image = pre_processing(
        image[: game_state.SCREENWIDTH, : int(game_state.BASEY)],
        opt.image_size,
        opt.image_size,
    )
    image = flow.Tensor(image, dtype=flow.float32)
    image = image.to("cuda")
    state = flow.cat(tuple(image for _ in range(4))).unsqueeze(0)

    replay_memory = []
    iter = 0
    # Step 4: run the game
    while iter < opt.num_iters:
        model.train()

        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
            (opt.num_iters - iter)
            * (opt.initial_epsilon - opt.final_epsilon)
            / opt.num_iters
        )
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            action = randint(0, 1)
        else:
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

        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(
            *batch
        )

        state_batch = flow.cat(tuple(state for state in state_batch))
        action_batch = flow.Tensor(
            np.array(
                [[1, 0] if action == 0 else [0, 1] for action in action_batch],
                dtype=np.float32,
            )
        )
        reward_batch = flow.Tensor(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = flow.cat(tuple(state for state in next_state_batch))

        state_batch = state_batch.to("cuda")
        action_batch = action_batch.to("cuda")
        reward_batch = reward_batch.to("cuda")
        next_state_batch = next_state_batch.to("cuda")
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = flow.cat(
            tuple(
                reward_batch[i]
                if terminal_batch[i]
                else reward_batch[i] + opt.gamma * flow.max(next_prediction_batch[i])
                for i in range(reward_batch.shape[0])
            )
        )

        q_value = flow.sum(current_prediction_batch * action_batch, dim=1)

        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        state = next_state
        iter += 1

        print(
            "Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                iter + 1,
                opt.num_iters,
                action,
                loss.numpy(),
                epsilon,
                reward,
                flow.max(prediction).numpy()[0],
            )
        )

        if (iter + 1) % 100000 == 0:
            flow.save(
                model.state_dict(),
                os.path.join(opt.save_checkpoint_path, "epoch_%d" % (iter + 1)),
            )
    flow.save(
        model.state_dict(),
        os.path.join(opt.save_checkpoint_path, "epoch_%d" % (iter + 1)),
    )
    print("train success!")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
