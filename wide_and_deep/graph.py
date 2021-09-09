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

###想把graph独立出来，后面需要改
class WideAndDeepGraph(flow.nn.Graph):
    def __init__(self, wdl_module,dataloader,bce_loss):
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
    def __init__(self, wdl_module,dataloader,bce_loss,optimizer):
        super(WideAndDeepTrainGraph, self).__init__(wdl_module,dataloader,bce_loss)
        self.add_optimizer(optimizer)

    def build(self):
        predicts, labels, loss = self.graph()
        loss.backward()
        return predicts, labels, loss