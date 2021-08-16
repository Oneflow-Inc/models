import oneflow as flow
import argparse
import numpy as np
import os
import time

import sys
sys.path.append(".")

from models.resnet50 import resnet50
from utils.ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
    parser = argparse.ArgumentParser("flags for train resnet50")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path", type=str, default="./ofrecord", help="dataset path"
    )
    parser.add_argument(
        "--ofrecord_part_num", type=int, default=1, help="ofrecord data part number"
    )
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--mom", type=float, default=0.9, help="momentum")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="train batch size"
    )
    parser.add_argument("--val_batch_size", type=int, default=32, help="val batch size")
    parser.add_argument(
        "--device_num", type=int, default=1, help=""
    )

    return parser.parse_args()


def main(args):

    rank = flow.distributed.get_rank()
    world_size = flow.distributed.get_world_size()

    device_list = [i for i in range(args.device_num)]

    placement = flow.placement("cpu", {0: device_list})
    sbp = [flow.sbp.split(0)]

    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=1281167,
        batch_size=args.train_batch_size * world_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="validation",
        dataset_size=50000,
        batch_size=args.val_batch_size * world_size,
        ofrecord_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )

    # oneflow init
    start_t = time.time()
    resnet50_module = resnet50()
    if args.load_checkpoint != "":
        print("load_checkpoint >>>>>>>>> ", args.load_checkpoint)
        resnet50_module.load_state_dict(flow.load(args.load_checkpoint))

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    of_cross_entropy = flow.nn.CrossEntropyLoss()

    placement = flow.placement("cuda", {0: device_list})
    sbp = [flow.sbp.broadcast]
    resnet50_module.to_consistent(placement=placement, sbp=sbp)
    of_cross_entropy.to_consistent(placement=placement, sbp=sbp)

    of_sgd = flow.optim.SGD(
        resnet50_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class Resnet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet50 = resnet50_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer("sgd", of_sgd)
            self.train_data_loader = train_data_loader
        
        def build(self):
            image, label = self.train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            logits = self.resnet50(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    resnet50_graph = Resnet50Graph()

    class Resnet50EvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet50 = resnet50_module
            self.val_data_loader = val_data_loader
        
        def build(self):
            image, label = self.val_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            with flow.no_grad():
                logits = self.resnet50(image)
                predictions = logits.softmax()
            return predictions, label

    resnet50_eval_graph = Resnet50EvalGraph()

    of_losses, of_accuracy = [], []
    all_samples = len(val_data_loader) * (args.val_batch_size * world_size)
    print_interval = 100


    for epoch in range(args.epochs):
        resnet50_module.train()

        for b in range(len(train_data_loader)):

            # oneflow graph train
            start_t = time.time()

            loss = resnet50_graph()

            end_t = time.time()
            if b % print_interval == 0:
                loss = loss.to_local()
                l = loss.numpy()
                of_losses.append(l)
                print(
                    "rank {} epoch {} train iter {} oneflow loss {}, train time : {}".format(
                        rank, epoch, b, l, end_t - start_t
                    )
                )

        print("rank %d epoch %d train done, start validation" % (rank, epoch))

        resnet50_module.eval()
        correct_of = 0.0
        for b in range(len(val_data_loader)):
            start_t = time.time()
            predictions, label = resnet50_eval_graph()
            
            predictions = predictions.to_consistent(sbp=[flow.sbp.broadcast])
            predictions = predictions.to_local()

            of_predictions = predictions.numpy()
            
            clsidxs = np.argmax(of_predictions, axis=1)

            label = label.to_consistent(sbp=[flow.sbp.broadcast])
            label = label.to_local()

            label_nd = label.numpy()
            
            for i in range(args.val_batch_size * world_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()

        top1 = correct_of / all_samples
        of_accuracy.append(top1)
        print("rank %d epoch %d, oneflow top1 val acc: %f" % (rank, epoch, top1))

        # if rank == 0:
            # get error
            # Traceback (most recent call last):
            # File "graph/train_consistent.py", line 203, in <module>
            #     main(args)
            # File "graph/train_consistent.py", line 182, in main
            #     flow.save(
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/check_point_v2.py", line 292, in save
            #     return _SaveVarDict(save_dir, obj)
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/check_point_v2.py", line 269, in _SaveVarDict
            #     for (_, _, slice) in _ReadSlice(var):
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/check_point_v2.py", line 200, in _ReadSlice
            #     yield from _ForEachSlice(container, ReadFromTensor)
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/check_point_v2.py", line 508, in _ForEachSlice
            #     yield (start_nd_idx, stop_nd_idx, f(container, start_nd_idx, stop_nd_idx))
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/check_point_v2.py", line 191, in ReadFromTensor
            #     return tensor[
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/tensor.py", line 91, in _getitem
            #     return flow.F.tensor_getitem(self, key)
            # File "/home/ldpe2g/oneFlow/oneflow/python/oneflow/framework/functional.py", line 26, in __call__
            #     return self.handle(*args, **kwargs)
            # oneflow._oneflow_internal.exception.UnimplementedException: 
            # File "/home/ldpe2g/oneFlow/oneflow/oneflow/core/functional/impl/array_functor.cpp", line 1152, in operator()
            #     x->device()
            # File "/home/ldpe2g/oneFlow/oneflow/oneflow/core/framework/tensor.h", line 448, in device
            # flow.save(
            #     resnet50_module.state_dict(),
            #     os.path.join(
            #         args.save_checkpoint_path,
            #         "epoch_%d_val_acc_%f" % (epoch, correct_of / all_samples),
            #     ),
            # )

    if rank == 0:
        writer = open("graph_of_losses.txt", "w")
        for o in of_losses:
            writer.write("%f\n" % o)
        writer.close()

        writer = open("graph/accuracy.txt", "w")
        for o in of_accuracy:
            writer.write("%f\n" % o)
        writer.close()

if __name__ == "__main__":
    args = _parse_args()
    main(args)
