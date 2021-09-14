# https://github.com/Kenneth111/TransformerDemo/blob/master/predict_odd_numbers.py
import sys
import argparse
import os
import shutil
import numpy as np

import oneflow as flow
import oneflow.nn as nn

sys.path.append("../")
from model import TransformerModel

TO_CUDA = True

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--vocab_sz", type=int, default=50000)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_encoder_layers", type=int, default=6)
parser.add_argument("--n_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=1024)

parser.add_argument("--load_dir", type=str, default=".")
parser.add_argument("--save_dir", type=str, default="./best_model")

args = parser.parse_args()


def to_cuda(tensor, flag=TO_CUDA, where="cuda"):
    if flag:
        return tensor.to(where)
    else:
        return tensor


def get_numbers(x, y, inp_len=3, out_len=3):
    data_x = np.array(
        [[x[i + j] for j in range(inp_len)] for i in range(len(x) - inp_len + 1)]
    )
    data_y = np.array(
        [[0] + [y[i + j] for j in range(out_len)] for i in range(len(y) - out_len + 1)]
    )  # 4997 * 3

    idx = np.arange(len(data_x))
    np.random.shuffle(idx)

    return data_x[idx], data_y[idx]


def shuffle_batch(data, label, batch_size):
    permu = np.random.permutation(len(data))
    data, label = data[permu], label[permu]
    batch_n = len(data) // batch_size

    x_batch = np.array(
        [data[i * batch_size : i * batch_size + batch_size] for i in range(batch_n)],
        dtype=np.int32,
    )
    y_batch = np.array(
        [label[i * batch_size : i * batch_size + batch_size] for i in range(batch_n)],
        dtype=np.int32,
    )

    return (
        flow.tensor(x_batch, dtype=flow.int64),
        flow.tensor(y_batch, dtype=flow.int64),
    )


def train(model, criterion, optimizer, train_x, train_y):
    model.train()
    epoch_loss = 0
    train_x, train_y = shuffle_batch(train_x, train_y, args.batch_size)
    for i, batch in enumerate(zip(train_x, train_y)):
        src, tgt = batch
        src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)
        src, tgt = to_cuda(src), to_cuda(tgt)
        last = tgt.shape[0]
        output = model(src, tgt[: last - 1, :])
        n = output.shape[-1]

        loss = criterion(output.permute(1, 2, 0), tgt[1:, :].permute(1, 0))
        loss.backward()

        optimizer.step()
        epoch_loss += loss.numpy()
        optimizer.zero_grad()
    return epoch_loss / train_x.shape[0]


def validation(model, criterion, val_x, val_y):
    model.eval()
    epoch_loss = 0
    val_x, val_y = shuffle_batch(val_x, val_y, args.batch_size)
    with flow.no_grad():
        for i, batch in enumerate(zip(val_x, val_y)):
            src, tgt = batch
            src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)
            src, tgt = to_cuda(src), to_cuda(tgt)
            last = tgt.shape[0]
            output = model(src, tgt[: last - 1, :])
            n = output.shape[-1]
            loss = criterion(output.permute(1, 2, 0), tgt[1:, :].permute(1, 0))
            epoch_loss += loss.numpy()
    return epoch_loss / val_x.shape[0]


def test(model, max_len=3, test_times=1, display=False):
    model.eval()
    res = []
    with flow.no_grad():
        for i in range(test_times):
            s = np.random.randint(1, 4998)
            cpu_src = [(s + j) * 2 for j in range(max_len)]
            src = to_cuda(flow.tensor(cpu_src, dtype=flow.int64).unsqueeze(1))
            tgt = [0] + [(s + j) * 2 + 1 for j in range(max_len)]
            pred = [0]
            flag = 1
            for j in range(max_len):
                inp = to_cuda(flow.tensor(pred, dtype=flow.int64).unsqueeze(1))
                output = model(src, inp)
                out_num = output.argmax(2)[-1].numpy()[0]
                pred.append(out_num)
                if pred[j + 1] != tgt[j + 1]:
                    flag = 0
            res.append(flag)
            if display:
                print("input: ", cpu_src)
                print("target: ", tgt)
                print("predict: ", pred)
    return sum(res) / test_times


def main():
    print("Generating data...", end="")
    voc_size = args.vocab_sz
    inp = np.arange(2, voc_size, 2)
    tgt = np.arange(3, voc_size, 2)
    data_x, data_y = get_numbers(inp, tgt)
    train_len = int(len(data_x) * 0.9)
    train_x, val_x = data_x[:train_len], data_x[train_len:]
    train_y, val_y = data_y[:train_len], data_y[train_len:]
    print("Done")

    print("Setting model...", end="")
    model = TransformerModel(
        input_sz=voc_size,
        output_sz=voc_size,
        d_model=args.d_model,
        nhead=args.n_head,
        num_encoder_layers=args.n_encoder_layers,
        num_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    if args.load_dir != ".":
        model.load_state_dict(flow.load(args.load_dir))
    model = to_cuda(model)
    criterion = to_cuda(nn.CrossEntropyLoss())

    optimizer = flow.optim.Adam(model.parameters(), lr=args.lr)
    print("Done")

    print("Training...")

    min_loss = 100
    for i in range(1, args.n_epochs + 1):
        epoch_loss = train(model, criterion, optimizer, train_x, train_y)
        epoch_loss_val = validation(model, criterion, val_x, val_y)
        print("epoch: {} train loss: {}".format(i, epoch_loss))
        print("epoch: {} val loss: {}".format(i, epoch_loss_val))
        if epoch_loss < min_loss:
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            else:
                shutil.rmtree(args.save_dir)
                assert not os.path.exists(args.save_dir)
                os.mkdir(args.save_dir)
            flow.save(model.state_dict(), args.save_dir)
        if i % 3 == 2:
            print(test(model, test_times=10))


if __name__ == "__main__":

    main()
