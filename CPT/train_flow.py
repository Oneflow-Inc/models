import os
import argparse
import shutil
import time

import numpy as np
import oneflow
from oneflow.utils.data import DataLoader
from classifier_flow import ClueAFQMCCPT
from dataset_flow import read_data, AFQMCDataset

time_map = {}


def get_afqmc_dataloader(task):

    train_input_ids, train_attention_mask, train_labels = read_data(task, "train")
    train_data = AFQMCDataset(train_input_ids, train_attention_mask, train_labels)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    eval_input_ids, eval_attention_mask, eval_labels = read_data(task, "eval")
    eval_data = AFQMCDataset(eval_input_ids, eval_attention_mask, eval_labels)
    evalloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False)

    return trainloader, evalloader


def get_acc(labels, logits, g):
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train_afqmc(args):

    if args.model_load_dir != ".":
        model.load_state_dict(oneflow.load(args.model_load_dir))

    criterion = oneflow.nn.CrossEntropyLoss().to(args.device)
    of_adam = oneflow.optim.Adam(model.parameters(), args.lr)

    time_map["t1"] = time.time()
    print("begin to load data...")
    start = time.time()
    trainloader, evalloader = get_afqmc_dataloader(args.task)
    print("Data Loading Time: %.2fs" % (time.time() - start))

    time_map["t5"] = time.time()
    print("begin to train...")
    for epoch in range(args.n_epochs):
        model.train()
        start = time.time()
        print("epoch{}:".format(epoch + 1))
        i = 0
        losses = 0
        train_length = len(trainloader.dataset)
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
            if i % 200 == 0:
                print(
                    "{:.2f}% loss={:.4f}".format(
                        i * args.batch_size / train_length * 100, losses / (i + 1)
                    )
                )
            of_adam.step()
            of_adam.zero_grad()
        print("Epoch %d training time %.2fs" % (epoch + 1, time.time() - start))

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
        print(
            "[Epoch:{0:d} ] accuracy: {1:.1f}%".format(
                epoch + 1, g["correct"] * 100 / g["total"]
            )
        )
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        else:
            shutil.rmtree(args.model_save_dir)
            assert not os.path.exists(args.model_save_dir)
            os.mkdir(args.model_save_dir)
        oneflow.save(model.state_dict(), args.model_save_dir)
    time_map["t6"] = time.time()
    print("Training Time: %.2fs" % (time_map["t6"] - time_map["t5"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--model_load_dir", type=str, default=".")
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument(
        "--pretrain_dir",
        type=str,
        default="/remote-home/share/shxing/cpt_pretrain_oneflow/cpt-base/",
    )
    parser.add_argument("--model_save_dir", type=str, default="cpt_pretrain_afqmc")
    parser.add_argument("--task", type=str, default="afqmc")
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()
    args.device = "cuda" if args.cuda else "cpu"
    args.is_train = True

    if args.task == "afqmc":
        args.n_classes = 2
        model = ClueAFQMCCPT(args.pretrain_dir, args.n_classes, args.is_train).to(
            args.device
        )
        train_afqmc(args)
    else:
        raise NotImplementedError
