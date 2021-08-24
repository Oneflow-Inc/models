import json
import os
import argparse

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim

from model.model import simple_CNN
from model.dataloader import create_batches_rnd


def get_args():
    parser = argparse.ArgumentParser("""Speaker Identification Demo Train""")
    parser.add_argument(
        "--label_dict", type=str, default="data_preprocessed/label_dict.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The number of images per batch"
    )
    parser.add_argument("--N_batches", type=int, default=100)
    parser.add_argument("--N_epoches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--wlen", type=int, default=3200)
    parser.add_argument("--fact_amp", type=float, default=0.2)
    parser.add_argument("--num_speakers", type=int, default=2)
    
    parser.add_argument("--output_path", type=str, default="save_models")

    args = parser.parse_args()
    return args


def train(opt):
    with open(opt.label_dict, "r") as f:
        lab_dict = json.load(f)

    cnn = simple_CNN(opt.num_speakers)
    cnn.to("cuda")

    cost = nn.CrossEntropyLoss()
    cost.to("cuda")

    optimizer = optim.RMSprop(cnn.parameters(), lr=opt.lr, alpha=opt.alpha, eps=opt.eps)

    output_folder = opt.output_path
    N_batches = opt.N_batches
    N_epoches = opt.N_epoches

    for epoch in range(N_epoches):
        cnn.train()

        loss_sum = 0
        err_sum = 0

        for i in range(N_batches):

            inp, lab = create_batches_rnd(
                lab_dict,
                batch_size=opt.batch_size,
                wlen=opt.wlen,
                fact_amp=opt.fact_amp,
                train=True,
            )
            inp = inp.unsqueeze(1)
            lab -= 1

            pout = cnn(inp)
            pred = flow.argmax(pout, dim=1)
            loss = cost(pout, lab.long())
            err = np.mean(pred.numpy() != lab.long().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_sum = loss_sum + loss.detach()
            err_sum = err_sum + err

        loss_tot = loss_sum / N_batches
        err_tot = err_sum / N_batches

        if epoch % 10 == 0:
            print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot.numpy(), err_tot))

    flow.save(cnn.state_dict(), os.path.join(output_folder, "CNN_model"))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
