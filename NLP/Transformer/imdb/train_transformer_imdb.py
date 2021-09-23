import numpy as np
import os
import argparse
import json
import time
import shutil

import oneflow as flow
import oneflow.nn as nn

from utils import pad_sequences, load_imdb_data
from model import TransformerEncoderModel

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--sequence_len", type=int, default=128)  # src_len
parser.add_argument("--vocab_sz", type=int, default=50000)  # emb_sz
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_encoder_layers", type=int, default=6)
parser.add_argument("--n_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=1024)

parser.add_argument("--imdb_path", type=str, default="../../imdb")
parser.add_argument("--load_dir", type=str, default=".")
parser.add_argument("--save_dir", type=str, default="./best_model")

args = parser.parse_args()
args.n_classes = 2  # tgt_len


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
        flow.tensor(x_batch, dtype=flow.int64).to("cuda"),
        flow.tensor(y_batch, dtype=flow.int64).to("cuda"),
    )


def prepare_data():

    print("Preparing data...")
    (train_data, train_labels), (test_data, test_labels) = load_imdb_data(
        args.imdb_path
    )

    with open(os.path.join(args.imdb_path, "word_index.json")) as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1

    train_data = pad_sequences(
        train_data, value=word_index["<PAD>"], padding="post", maxlen=args.sequence_len
    )
    test_data = pad_sequences(
        test_data, value=word_index["<PAD>"], padding="post", maxlen=args.sequence_len
    )

    return train_data, train_labels, test_data, test_labels


def acc(labels, logits, g):

    predictions = np.argmax(logits.numpy(), 1)
    right_count = np.sum(predictions == labels.numpy())
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train():

    train_data, train_labels, test_data, test_labels = prepare_data()

    best_accuracy = 0.0
    best_epoch = 0

    print("Setting model...")
    model = TransformerEncoderModel(
        emb_sz=args.vocab_sz,
        n_classes=args.n_classes,
        d_model=args.d_model,
        nhead=args.n_head,
        num_encoder_layers=args.n_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_first=True,
    )
    criterion = nn.CrossEntropyLoss()
    model.to("cuda")
    criterion.to("cuda")
    of_adam = flow.optim.Adam(model.parameters(), lr=args.lr)

    if args.load_dir != ".":
        model.load_state_dict(flow.load(args.load_dir))

    print("Starting training...")
    training_time = 0
    for epoch in range(1, args.n_epochs + 1):
        print("[Epoch:{}]".format(epoch))
        model.train()
        data, label = shuffle_batch(train_data, train_labels, args.batch_size)
        s_t = time.time()
        epoch_loss = 0
        for i, (texts, labels) in enumerate(zip(data, label)):
            output = model(texts)
            loss = criterion(output, labels)
            loss.backward()
            of_adam.step()
            of_adam.zero_grad()
            epoch_loss += loss.numpy()
            if i % 50 == 0 or i == data.shape[0] - 1:
                print(
                    "{0:d}/{1:d}, loss:{2:.4f}".format(
                        i + 1, data.shape[0], loss.numpy()
                    )
                )
        epoch_loss /= data.shape[0]
        e_t = time.time() - s_t
        training_time += e_t
        print(
            "Epoch:{0:d} training time:{1:.2f}s, loss:{2:.4f}".format(
                epoch, e_t, epoch_loss
            )
        )

        model.eval()
        data, label = shuffle_batch(test_data, test_labels, args.batch_size)
        g = {"correct": 0, "total": 0}
        for i, (texts, labels) in enumerate(zip(data, label)):
            logits = model(texts)
            acc(labels, logits, g)

        accuracy = g["correct"] * 100 / g["total"]
        print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(epoch, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            else:
                shutil.rmtree(args.save_dir)
                assert not os.path.exists(args.save_dir)
                os.mkdir(args.save_dir)
            print("Epoch:{} save best model.".format(best_epoch))
            flow.save(model.state_dict(), args.save_dir)

    print(
        "Epoch:{} get best accuracy:{}, average training time:{}s".format(
            best_epoch, best_accuracy, training_time / args.n_epochs
        )
    )


if __name__ == "__main__":

    train()
