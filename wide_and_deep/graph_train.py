import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import oneflow as flow

from config import get_args
from dataloader_utils import OFRecordDataLoader
from wide_and_deep_module import WideAndDeep
from util import dump_to_npy, save_param_npy
from eager_train import prepare_modules, print_eval_metrics

if __name__ == '__main__':
    args = get_args()

    train_dataloader, val_dataloader, wdl_module, bce_loss, opt = prepare_modules(
        args)

    class WideAndDeepGraph(flow.nn.Graph):
        def __init__(self, dataloader):
            super(WideAndDeepGraph, self).__init__()
            self.module = wdl_module
            self.dataloader = dataloader
            self.bce_loss = bce_loss

        def build(self):
            with flow.no_grad():
                return self.graph()

        def graph(self):
            labels, dense_fields, wide_sparse_fields, deep_sparse_fields = self.dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")

            predicts = self.module(
                dense_fields, wide_sparse_fields, deep_sparse_fields)
            loss = self.bce_loss(predicts, labels)
            return predicts, labels, loss

    class WideAndDeepTrainGraph(WideAndDeepGraph):
        def __init__(self, dataloader):
            super(WideAndDeepTrainGraph, self).__init__(dataloader)
            self.add_optimizer(opt)

        def build(self):
            predicts, labels, loss = self.graph()
            loss.backward()
            return predicts, labels, loss

    eval_graph = WideAndDeepGraph(val_dataloader)
    train_graph = WideAndDeepTrainGraph(train_dataloader)

    losses = []
    wdl_module.train()

    for i in range(args.max_iter):
        predicts, labels, train_loss = train_graph()
        losses.append(train_loss.numpy().mean())

        if (i+1) % args.print_interval == 0:
            l = sum(losses) / len(losses)
            losses = []
            print(f"iter {i} train_loss {l} time {time.time()}")
            if args.eval_batchs <= 0:
                continue

            eval_loss_acc = 0.0
            lables_list = []
            predicts_list = []
            wdl_module.eval()
            for j in range(args.eval_batchs):
                predicts, labels, eval_loss = eval_graph()

                eval_loss_acc += eval_loss.numpy().mean()
                lables_list.append(labels.numpy())
                predicts_list.append(predicts.numpy())

            print_eval_metrics(eval_loss_acc/args.eval_batchs,
                               lables_list, predicts_list)
            wdl_module.train()
