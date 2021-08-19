import oneflow as flow

import numpy as np
import time
import argparse
import torch

import models.ghostnet_torch as pytorch_ghostnet
from models.ghostnet import ghostnet


def _parse_args():
    parser = argparse.ArgumentParser("flags for compare oneflow and pytorch speed")
    return parser.parse_args()


def main(args):

    batch_size = 16
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)

    ghostnet_module = ghostnet()
    # set for eval mode
    # ghostnet_module.eval()
    image = flow.tensor(image_nd)
    label = flow.tensor(label_nd)
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    image_gpu = image.to("cuda")
    label = label.to("cuda")
    ghostnet_module.to("cuda")
    corss_entropy.to("cuda")

    learning_rate = 0.001
    mom = 0.9
    of_sgd = flow.optim.SGD(ghostnet_module.parameters(), lr=learning_rate, momentum=mom)

    bp_iters = 500
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0
    all_of_losses = []
    all_torch_losses = []

    print("start oneflow training loop....")
    start_t = time.time()
    for i in range(bp_iters):
        s_t = time.time()
        logits = ghostnet_module(image_gpu)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        of_sgd.step()
        of_sgd.zero_grad()
        update_time += time.time() - s_t
        all_of_losses.append(float(loss.numpy()))

    of_loss = loss.numpy()
    end_t = time.time()

    print("oneflow traning loop avg time : {}".format((end_t - start_t) / bp_iters))
    print("forward avg time : {}".format(for_time / bp_iters))
    print("backward avg time : {}".format(bp_time / bp_iters))
    print("update parameters avg time : {}".format(update_time / bp_iters))

    #####################################################################################################
    # pytorch ghostnet
    torch_ghostnet_module = pytorch_ghostnet.ghostnet()

    # set for eval mode
    # torch_ghostnet_module.eval()
    torch_ghostnet_module.to("cuda")
    torch_sgd = torch.optim.SGD(
        torch_ghostnet_module.parameters(), lr=learning_rate, momentum=mom
    )

    image = torch.tensor(image_nd)
    image_gpu = image.to("cuda")
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to("cuda")
    label = torch.tensor(label_nd, dtype=torch.long).to("cuda")

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    print("start pytorch training loop....")
    start_t = time.time()
    for i in range(bp_iters):
        s_t = time.time()
        logits = torch_ghostnet_module(image_gpu)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        torch_sgd.step()
        torch_sgd.zero_grad()
        update_time += time.time() - s_t
        all_torch_losses.append(float(loss.cpu().detach().numpy()))

    torch_loss = loss.cpu().detach().numpy()
    end_t = time.time()
    print("pytorch traning loop avg time : {}".format((end_t - start_t) / bp_iters))
    print("forward avg time : {}".format(for_time / bp_iters))
    print("backward avg time : {}".format(bp_time / bp_iters))
    print("update parameters avg time : {}".format(update_time / bp_iters))

    np.savetxt('all_of_losses.txt',all_of_losses)
    np.savetxt('all_torch_losses.txt',all_torch_losses)

if __name__ == "__main__":
    args = _parse_args()
    main(args)
