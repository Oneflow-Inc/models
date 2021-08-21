import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

import numpy as np
import oneflow as flow

from oneflow_gpt.config import get_args
from oneflow_gpt import distribute as dist
from oneflow_gpt.data import GPTDataLoader
from oneflow_gpt.model import GPTModel, ParallelSparseSoftmaxCrossEntropyLoss


class GPTTrainGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.data_loader = GPTDataLoader()
        self.gpt_model = GPTModel()
        self.sparse_softmax_cross_entroy_loss = ParallelSparseSoftmaxCrossEntropyLoss()

    def build(self):
        data, label = self.data_loader()
        logits = self.gpt_model(data)
        loss = self.sparse_softmax_cross_entroy_loss(logits, label)
        return loss


def train():
    args = get_args()
    dist_util = dist.get_dist_util()

    # if dist_util.model_parallel_size > 1:
    #     flow.config.nccl_use_compute_stream(True)

    if args.use_rdma:
        flow.config.use_rdma(True)

    trainer = GPTTrainGraph()

    print("Training...")
    loss = trainer()
    print("loss:", loss.to_local().numpy())

    # iteration = snapshot.iter
    # while iteration < args.train_iters:
    #     trainer()
    #     # snapshot.step()
    #     iteration = snapshot.iter


if __name__ == "__main__":
    train()
