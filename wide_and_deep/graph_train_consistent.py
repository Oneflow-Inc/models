import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import oneflow as flow

from config import get_args
from dataloader_utils_consistent import OFRecordDataLoader
from wide_and_deep_module import WideAndDeep
from util import dump_to_npy, save_param_npy
from eager_train_consistent import prepare_modules, print_eval_metrics

world_size = flow.env.get_world_size()
placement = flow.placement("cpu", {0: range(world_size)})

if __name__ == "__main__":
    args = get_args()

    train_dataloader, val_dataloader, wdl_module, bce_loss, opt = prepare_modules(args)

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
            (
                labels,
                dense_fields,
                wide_sparse_fields,
                deep_sparse_fields,
            ) = self.dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.module(dense_fields, wide_sparse_fields, deep_sparse_fields)
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

        #train_loss是partial_sum
        train_loss = train_loss / world_size
        #各个rank 打印local loss
        losses.append(train_loss.to_local().numpy().mean())

        if (i + 1) % args.print_interval == 0:
            l = sum(losses) / len(losses)
            losses = []
            rank=flow.env.get_rank()
            print(f"device {rank}: iter {i+1} train_loss {l} time {time.time()}")
            if args.eval_batchs <= 0:
                continue

            eval_loss_acc = 0.0
            lables_list = []
            predicts_list = []
            wdl_module.eval()
            for j in range(args.eval_batchs):
                predicts, labels, eval_loss = eval_graph()

                #train_loss是partial_sum
                eval_loss = eval_loss / world_size
                #各个rank 打印local loss
                losses.append(eval_loss.to_local().numpy().mean())
                eval_loss_acc += eval_loss.to_local().numpy().mean()
                lables_list.append(labels.to_local().numpy())
                predicts_list.append(predicts.to_local().numpy())

            print_eval_metrics(
                i + 1, eval_loss_acc / args.eval_batchs, lables_list, predicts_list
            )
            wdl_module.train()
