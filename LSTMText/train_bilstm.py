import os
import argparse
import time
import json

import numpy as np
import oneflow as flow
import oneflow.nn as nn
import shutil

from model import LSTMText
from utils import pad_sequences, load_imdb_data, colored_string

time_map = {}


def shuffle_batch(data, label, batch_size):
    permu = np.random.permutation(len(data))
    data, label = data[permu], label[permu]
    batch_n = len(data) // batch_size
    x_batch = np.array([data[i * batch_size:i * batch_size + batch_size]
                        for i in range(batch_n)], dtype=np.int32)
    y_batch = np.array([label[i * batch_size:i * batch_size + batch_size]
                        for i in range(batch_n)], dtype=np.int32)
    x_batch = flow.Tensor(x_batch, dtype=flow.int32).to('cuda')
    y_batch = flow.Tensor(y_batch, dtype=flow.int32).to('cuda')
    return x_batch, y_batch


def load_data():
    print(colored_string('Start Loading Data', 'green'))
    start = time.time()
    (train_data, train_labels), (test_data, test_labels) = load_imdb_data(args.imdb_path)
    with open(os.path.join(args.imdb_path, 'word_index.json')) as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<UNK>"] = 1
    train_data = pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)
    test_data = pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)
    print(colored_string('Data Loading Time: %.2fs' %
                         (time.time() - start), 'blue'))
    return train_data, train_labels, test_data, test_labels


def get_acc(labels, logits, g):
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train_eager(args):
    time_map['t1'] = time.time()
    train_data, train_labels, test_data, test_labels = load_data()
    train_data = train_data[0:25000, :]
    train_labels = train_labels[0:25000]
    test_data = test_data[0:25000, :]
    test_labels = test_labels[0:25000]
    
    model_eager = LSTMText(args.emb_num, args.emb_dim, hidden_size=args.hidden_size,
                           nfc=args.nfc, n_classes=args.n_classes, batch_size=args.batch_size)
    if args.model_load_dir != ".":
        model_eager.load_state_dict(flow.load(args.model_load_dir))
    print(colored_string("Start Training in Eager Mode", 'green'))
    time_map['t5'] = time.time()
    criterion = flow.nn.CrossEntropyLoss()
    model_eager.to('cuda')
    criterion.to('cuda')
    
    of_adam = flow.optim.Adam(model_eager.parameters(), args.lr)
    
    for epoch in range(1, args.n_epochs + 1):
        print("[Epoch:{}]".format(epoch))
        start = time.time()
        model_eager.train()
        data, label = shuffle_batch(train_data, train_labels, args.batch_size)
        i = 0
        losses = 0
        len_data = data.shape[0]
        for i in range(len_data):
            texts = data[i]
            labels = label[i]
            output = model_eager(texts)
            loss = criterion(output, labels.reshape(args.batch_size))
            losses += loss.numpy()
            loss.backward()
            of_adam.step()
            of_adam.zero_grad()
            if i % 50 == 0:
                print("batch", i, "loss", losses / (i + 1))
        print("Epoch %d training time %.2fs" % (epoch, time.time() - start))
        
        if epoch % args.model_save_every_n_epochs == 0:
            model_eager.eval()
            data, label = shuffle_batch(test_data, test_labels, args.batch_size)
            g = {"correct": 0, "total": 0}
            for i, (texts, labels) in enumerate(zip(data, label)):
                logits = model_eager(texts)
                logits = logits.numpy()
                labels = labels.numpy()
                get_acc(labels, logits, g)
            print(g["correct"], g["total"])
            print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(
                epoch, g["correct"] * 100 / g["total"]))
            if not os.path.exists(args.model_save_dir):
                os.mkdir(args.model_save_dir)
            else:
                shutil.rmtree(args.model_save_dir)
                assert not os.path.exists(args.model_save_dir)
                os.mkdir(args.model_save_dir)
            flow.save(model_eager.state_dict(), args.model_save_dir)
    
    time_map['t6'] = time.time()
    print(colored_string("Training Time: %.2fs" %
                         (time_map['t6'] - time_map['t5']), 'blue'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--nfc', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_load_dir', type=str, default='.')
    parser.add_argument('--model_save_every_n_epochs', type=int, default=5)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--model_save_dir', type=str, default='./save')
    parser.add_argument('--imdb_path', type=str, default='../imdb')
    
    args = parser.parse_args()
    args.emb_num = 50000
    args.n_classes = 2
    
    train_eager(args)
