import oneflow.experimental as flow

import numpy as np
import time
import argparse
import torch
import string

from models.rnn_model_pytorch import RNN_PYTORCH
from models.rnn_model import RNN

# shared hyperparameters
n_hidden = 5000
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_categories = 25600
learning_rate = 0.0005


def _parse_args():
    parser = argparse.ArgumentParser("flags for compare oneflow and pytorch speed")
    return parser.parse_args()


def letterToIndex(letter):
    return all_letters.find(letter)


def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    rnn_module = RNN(n_letters, n_hidden, n_categories)
    # Fake data, only for speed test purpose
    test_word = "Depeng"
    category_tensor = flow.Tensor([1], dtype=flow.int64)
    line_tensor = flow.Tensor(len(test_word), 1, n_letters)
    flow.nn.init.zeros_(line_tensor)
    for li, letter in enumerate(test_word):
        line_tensor[li, 0, letterToIndex(letter)] = 1
    criterion = flow.nn.NLLLoss()

    category_tensor_gpu = category_tensor.to("cuda")
    line_tensor_gpu = line_tensor.to("cuda")
    rnn_module.to("cuda")
    criterion.to("cuda")
    of_sgd = flow.optim.SGD(rnn_module.parameters(), lr=learning_rate)

    bp_iters = 50
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    print("start oneflow training loop....")
    start_t = time.time()
    for i in range(bp_iters):
        s_t = time.time()
        hidden = rnn_module.initHidden()
        for j in range(line_tensor_gpu.size()[0]):
            output, hidden = rnn_module(line_tensor_gpu[j], hidden)
        loss = criterion(output, category_tensor_gpu)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        of_sgd.step()
        of_sgd.zero_grad()
        update_time += time.time() - s_t

    of_loss = loss.numpy()
    end_t = time.time()

    print("oneflow traning loop avg time : {}".format((end_t - start_t) / bp_iters))
    print("forward avg time : {}".format(for_time / bp_iters))
    print("backward avg time : {}".format(bp_time / bp_iters))
    print("update parameters avg time : {}".format(update_time / bp_iters))

    #####################################################################################################
    # # pytorch RNN
    torch_rnn_module = RNN_PYTORCH(n_letters, n_hidden, n_categories)

    torch_rnn_module.to("cuda")

    category_tensor = torch.tensor([1], dtype=torch.long)
    line_tensor = torch.zeros(len(test_word), 1, n_letters)
    for li, letter in enumerate(test_word):
        line_tensor[li][0][letterToIndex(letter)] = 1
    criterion = torch.nn.NLLLoss()

    category_tensor_gpu = category_tensor.to("cuda")
    line_tensor_gpu = line_tensor.to("cuda")
    torch_rnn_module.to("cuda")
    criterion.to("cuda")

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    print("start pytorch training loop....")
    start_t = time.time()
    for i in range(bp_iters):
        s_t = time.time()
        hidden = torch_rnn_module.initHidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = torch_rnn_module(line_tensor_gpu[i], hidden)
        loss = criterion(output, category_tensor_gpu)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        for p in torch_rnn_module.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)
        torch_rnn_module.zero_grad()
        update_time += time.time() - s_t

    torch_loss = loss.cpu().detach().numpy()
    end_t = time.time()
    print("pytorch traning loop avg time : {}".format((end_t - start_t) / bp_iters))
    print("forward avg time : {}".format(for_time / bp_iters))
    print("backward avg time : {}".format(bp_time / bp_iters))
    print("update parameters avg time : {}".format(update_time / bp_iters))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
