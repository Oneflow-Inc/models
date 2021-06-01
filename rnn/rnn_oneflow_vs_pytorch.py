import oneflow.experimental as flow

import numpy as np
import time
import argparse
import torch
import string

from models.rnn_model_pytorch import RNN_PYTORCH
from models.rnn_model import RNN

#shared hyperparameters
n_hidden = 128
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
n_categories = 256

def _parse_args():
    parser = argparse.ArgumentParser("flags for compare oneflow and pytorch speed")
    return parser.parse_args()

def letterToIndex(letter):
    return all_letters.find(letter)

def lineToTensor(line):
    tensor = flow.Tensor(len(line), n_letters)
    flow.nn.init.zeros_(tensor)
    for li, letter in enumerate(line):
        # NOTE(Liang Depeng): oneflow Tensor does not support tensor[li][letterToIndex(letter)] = 1
        tensor[li, letterToIndex(letter)] = 1
    return tensor.to("cuda")

def lineToTensor_torch(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    rnn_module = RNN(n_letters, n_hidden, n_categories)
    # set for eval mode
    # res50_module.eval()
    category_tensor = flow.Tensor([1], dtype=flow.int64)
    line_tensor = lineToTensor('Depeng')
    cross_entropy = flow.nn.CrossEntropyLoss()

    category_tensor_gpu = category_tensor.to('cuda')
    line_tensor_gpu = line_tensor.to('cuda')
    rnn_module.to('cuda')
    cross_entropy.to('cuda')

    learning_rate = 0.0005
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
        loss = cross_entropy(output, category_tensor_gpu)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t
        
        s_t = time.time()
        for p in rnn_module.parameters():
            p[:] = p - learning_rate * p.grad
        for p in rnn_module.parameters():
            p.grad.fill_(0)
        update_time += time.time() - s_t

    of_loss = loss.numpy()
    end_t = time.time()

    print('oneflow traning loop avg time : {}'.format((end_t - start_t) / bp_iters))
    print('forward avg time : {}'.format(for_time / bp_iters))
    print('backward avg time : {}'.format(bp_time / bp_iters))
    print('update parameters avg time : {}'.format(update_time / bp_iters))

    #####################################################################################################
    # # pytorch resnet50
    torch_rnn_module = RNN_PYTORCH(n_letters, n_hidden, n_categories)

    # set for eval mode
    # torch_res50_module.eval()
    torch_rnn_module.to('cuda')

    category_tensor = torch.tensor([1], dtype=torch.long)
    line_tensor = lineToTensor_torch('Depeng')
    cross_entropy = torch.nn.CrossEntropyLoss()
    
    category_tensor_gpu = category_tensor.to('cuda')
    line_tensor_gpu = line_tensor.to('cuda')
    torch_rnn_module.to('cuda')
    cross_entropy.to('cuda')

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
        loss = cross_entropy(output, category_tensor_gpu)
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
    print('pytorch traning loop avg time : {}'.format((end_t - start_t) / bp_iters))
    print('forward avg time : {}'.format(for_time / bp_iters))
    print('backward avg time : {}'.format(bp_time / bp_iters))
    print('update parameters avg time : {}'.format(update_time / bp_iters))

if __name__ == "__main__":
    args = _parse_args()
    main(args)