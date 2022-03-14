import oneflow as flow
import os
import sys


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from config import get_args
from eager_test import Tester
from graph_train import DLRMValGraph


class GraphTester(Tester):
    def __init__(self, args):
        super(GraphTester, self).__init__(args)
        self.eval_graph = DLRMValGraph(self.dlrm_module, args.use_fp16)

    def inference(self, dense_fields, sparse_fields):
        return self.eval_graph(dense_fields, sparse_fields)


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    tester = GraphTester(args)
    tester.test()
