import argparse

import time
import oneflow as flow
from oneflow.utils.data import DataLoader

import numpy as np
import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel as t_GPT2LMHeadModel,
    GPT2Config as t_GPT2Config,
)

import model as mnn
import pt_model as pnn

from model_config import GPT2Config
from model import GPT2LMHeadModel
from pt_model import GPT2LMHeadModel as pt_GPT2LMHeadModel
from trainer import Trainer
from gpt_dataset import GPTDataset
from tokenizer import build_tokenizer


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset",
        required=False,
        type=str,
        default="data/corpus.small",
        help="train dataset",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="data/corpus.small",
        help="test set for evaluation",
    )
    parser.add_argument("--vocab_file", required=False, default="vocab.json", type=str)
    parser.add_argument("--merges_file", required=False, default="merge.txt", type=str)
    parser.add_argument(
        "--output_path",
        required=False,
        default="output/model",
        type=str,
        help="save path",
    )

    parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument(
        "--batch_size", type=int, default=4, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument(
        "--num_workers", type=int, default=0, help="dataloader worker size"
    )

    parser.add_argument(
        "--with_cuda",
        type=bool,
        default=True,
        help="training with CUDA: true, or false",
    )

    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam first beta value"
    )

    args = parser.parse_args()

    print("building tokenizer")
    tokenizer = build_tokenizer(
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        tokenizer_type="GPT2BPETokenizer",
    )

    print("building train dataset")
    train_dataset = GPTDataset(args.train_dataset, tokenizer, args.seq_len)

    print("building train dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    for i, b in enumerate(train_data_loader):
        if i == 2:
            batch = b
            break

    of_batch = batch.cuda()

    print("building model")
    config = GPT2Config()

    pt_batch = torch.from_numpy(batch.numpy()).long().cuda()

    model = pt_GPT2LMHeadModel(config)

    model.load_state_dict(torch.load("gpt2_model.pt"))
    model.lm_head.weight = model.transformer.wte.weight

    model.cuda()
    model.eval()

    learning_rate = 0.01
    mom = 0.9
    pt_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0
    pt_loss = list()
    loss = None
    print("start pytorch training loop....")
    start_t = time.time()
    for epoch in range(args.epochs):
        s_t = time.time()
        loss = model(pt_batch, labels=pt_batch)[0]
        for_time += time.time() - s_t

        pt_loss.append(loss.item())

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        pt_optimizer.step()
        pt_optimizer.zero_grad()
        update_time += time.time() - s_t

    end_t = time.time()

    print("pytorch traning loop avg time : {}".format((end_t - start_t) / args.epochs))
    print("forward avg time : {}".format(for_time / args.epochs))
    print("backward avg time : {}".format(bp_time / args.epochs))
    print("update parameters avg time : {}".format(update_time / args.epochs))

    pt_parameters_names = []
    pt_parameters_value = []
    for name, param in model.named_parameters():
        pt_parameters_names.append(name)
        pt_parameters_value.append(param.cpu().detach().numpy())

    model = GPT2LMHeadModel(config)

    model.load_state_dict(flow.load("gpt2_oneflow_model"))
    model.lm_head.weight = model.transformer.wte.weight

    model.cuda()
    model.eval()

    optimizer = flow.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )

    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0
    of_loss = list()

    print("start oneflow training loop....")
    start_t = time.time()
    for epoch in range(args.epochs):
        s_t = time.time()
        loss = model(of_batch, labels=of_batch)[0]
        for_time += time.time() - s_t

        of_loss.append(loss.numpy())

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        optimizer.step()
        optimizer.zero_grad()
        update_time += time.time() - s_t

    end_t = time.time()

    print("oneflow traning loop avg time : {}".format((end_t - start_t) / args.epochs))
    print("forward avg time : {}".format(for_time / args.epochs))
    print("backward avg time : {}".format(bp_time / args.epochs))
    print("update parameters avg time : {}".format(update_time / args.epochs))

    for i in range(args.epochs):
        print(i, of_loss[i], pt_loss[i])

    import matplotlib.pyplot as plt

    plt.switch_backend("agg")
    epochs = np.arange(1, args.epochs + 1)

    plt.plot(epochs, of_loss, label="oneflow")
    plt.plot(epochs, pt_loss, label="pytorch")
    plt.legend()
    plt.savefig("./1.jpg")
    plt.show()


if __name__ == "__main__":
    main()
