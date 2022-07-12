import os
import argparse
import shutil
import time

import numpy as np
import oneflow
from oneflow.utils.data import DataLoader
from classifier_MNLI import MNLIRoBERTa
from SST2Dataset import read_data, SST2Dataset
from config import train_config

time_map = {}
def get_acc(labels, logits, g):
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train(args):
    model = MNLIRoBERTa(args.pretrain_dir, args.kwargs_path,
                        args.roberta_hidden_size, args.n_classes, args.is_train).to(args.device)
    criterion = oneflow.nn.CrossEntropyLoss().to(args.device)
    of_adam = oneflow.optim.Adam(model.parameters(), args.lr)

    time_map['t1'] = time.time()
    print("begin to load data...")
    start = time.time()
    input_ids, attention_mask, labels = read_data('train')
    train_data = SST2Dataset(input_ids, attention_mask, labels)
    trainloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    input_ids, attention_mask, labels = read_data('eval')
    eval_data = SST2Dataset(input_ids, attention_mask, labels)
    evalloader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=True)
    print('Data Loading Time: %.2fs' % (time.time() - start))
    time_map['t5'] = time.time()
    print("begin to train...")
    for epoch in range(args.n_epochs):
        model.train()
        start = time.time()
        print("epoch{}:".format(epoch+1))
        i = 0
        losses = 0
        for iter in trainloader:
            input_ids, attention_mask, labels = iter
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            i += 1
            output = model(input_ids, attention_mask)
            labels = labels.reshape(-1).to(args.device)
            loss = criterion(output, labels)
            losses += loss.detach()
            loss.backward()
            of_adam.step()
            of_adam.zero_grad()
            if i % 50 == 0:
                print("batch", i, "loss", losses / (i + 1))
        print("Epoch %d training time %.2fs" % (epoch+1, time.time() - start))

        model.eval()
        g = {"correct": 0, "total": 0}
        for iter in evalloader:
            input_ids, attention_mask, labels = iter
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            labels = labels.to(args.device)
            labels = labels.reshape(-1).to(args.device)
            logits = model(input_ids, attention_mask)
            logits = logits.numpy()
            labels = labels.numpy()
            get_acc(labels, logits, g)
        print("correct / total: {}/ {}".format(g["correct"], g["total"]))
        print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(
            epoch+1, g["correct"] * 100 / g["total"]))
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        else:
            shutil.rmtree(args.model_save_dir)
            assert not os.path.exists(args.model_save_dir)
            os.mkdir(args.model_save_dir)
        oneflow.save(model.state_dict(), args.model_save_dir)
    time_map['t6'] = time.time()
    print("Training Time: %.2fs" % (time_map['t6'] - time_map['t5']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_load_dir', type=str, default='.')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--kwargs_path', type=str,
                        default='./roberta-base-oneflow/parameters.json')
    parser.add_argument('--pretrain_dir', type=str,
                        default='./roberta-base-oneflow/weights')
    parser.add_argument('--model_save_dir', type=str,
                        default='./pretrain_model_MNLI')
    parser.add_argument('--task', type=str,
                        default='MNLI')

    args = parser.parse_args()
    args.device = 'cuda' if oneflow.cuda.is_available() else 'cpu'
    train_config(args)
    train(args)
