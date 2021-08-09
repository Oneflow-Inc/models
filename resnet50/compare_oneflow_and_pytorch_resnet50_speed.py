import oneflow as flow

import numpy as np
import time
import argparse
import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50


def _parse_args():
    parser = argparse.ArgumentParser("flags for compare oneflow and pytorch speed")
    return parser.parse_args()


def main(args):

    batch_size = 16
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)

    res50_module = resnet50()
    # set for eval mode
    # res50_module.eval()
    image = flow.tensor(image_nd)
    label = flow.tensor(label_nd)
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    image_gpu = image.to("cuda")
    label = label.to("cuda")
    res50_module.to("cuda")
    corss_entropy.to("cuda")

    learning_rate = 0.01
    mom = 0.9
    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)

    bp_iters = 50
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    print("start oneflow training loop....")
    start_t = time.time()
    for i in range(bp_iters):
        s_t = time.time()
        logits = res50_module(image_gpu)
        loss = corss_entropy(logits, label)
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
    # pytorch resnet50
    torch_res50_module = pytorch_resnet50.resnet50()

    # set for eval mode
    # torch_res50_module.eval()
    torch_res50_module.to("cuda")
    torch_sgd = torch.optim.SGD(
        torch_res50_module.parameters(), lr=learning_rate, momentum=mom
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
        logits = torch_res50_module(image_gpu)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        torch_sgd.step()
        torch_sgd.zero_grad()
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
