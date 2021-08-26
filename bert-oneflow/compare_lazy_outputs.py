
import oneflow.nn as nn
import oneflow as flow
from modeling import BertForPreTraining


if __name__ == "__main__":
    lazy_state_dict = flow.load("../../OneFlow-Benchmark/LanguageModeling/BERT/snapshots/snapshot_snapshot_1")

    bert_module = BertForPreTraining(
        30522,
        128,
        768,
        12,
        12,
        3072,
        nn.GELU(),
        0.1,
        0.1,
        512,
        2,
    )

    eager_state_dict = bert_module.state_dict()

