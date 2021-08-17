import oneflow as flow
import argparse
import numpy as np
import os
import time
from oneflow.nn.module import Module
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

    parser.add_argument("--num_classes", type=int, default=1000, help="number of class")
    parser.add_argument("--label_smooth_rate", type=float, default=0.0, help="val batch size")


    return parser.parse_args()

class LabelSmoothLoss(Module):
    def __init__(self, num_classes=-1, smooth_rate=0.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth_rate = smooth_rate
    
    def forward(self, input, label):
        print(input.shape, input.dtype)
        print(label.shape, label.dtype)
        print("===================")
        onehot_label = flow.F.one_hot(label, num_classes= self.num_classes, 
                                                on_value=1-self.smooth_rate, 
                                                off_value=self.smooth_rate/(self.num_classes-1))
        log_prob = input.softmax(dim=-1).log()
        onehot_label = onehot_label.to(log_prob.dtype).to("cuda")
        loss = flow.mul(log_prob * -1, onehot_label).sum(dim=-1).mean()
        return loss

def main(args):
    train_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="train",
        dataset_size=9469,
        batch_size=args.train_batch_size,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=3925,
        batch_size=args.val_batch_size,
    )

    # oneflow init
    start_t = time.time()
    resnet50_module = resnet50()
    if args.load_checkpoint != "":
        print("load_checkpoint >>>>>>>>> ", args.load_checkpoint)
        resnet50_module.load_state_dict(flow.load(args.load_checkpoint))

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    # of_cross_entropy = flow.nn.CrossEntropyLoss()
    of_loss = LabelSmoothLoss(num_classes=args.num_classes, smooth_rate=args.label_smooth_rate)

    resnet50_module.to("cuda")
    of_loss.to("cuda")

    of_sgd = flow.optim.SGD(
        resnet50_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class Resnet50Graph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet50 = resnet50_module
            self.of_loss = of_loss
            # self.add_optimizer("sgd", of_sgd)
            self.add_optimizer(of_sgd)
            self.train_data_loader = train_data_loader
        
        def build(self):
            image, label = self.train_data_loader()
            image = image.to("cuda")
            logits = self.resnet50(image)
            loss = self.of_loss(logits, label)
            loss.backward()
            return loss

    resnet50_graph = Resnet50Graph()

    class Resnet50EvalGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet50 = resnet50_module
            self.val_data_loader = val_data_loader
        
        def build(self,):
            image, label = self.val_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            with flow.no_grad():
                logits = self.resnet50(image)
                predictions = logits.softmax()
            return predictions, label

    resnet50_eval_graph = Resnet50EvalGraph()

    of_losses, of_accuracy = [], []
    all_samples = len(val_data_loader) * args.val_batch_size
    print_interval = 100


    for epoch in range(args.epochs):
        resnet50_module.train()

        for b in range(len(train_data_loader)):
            # oneflow graph train
            start_t = time.time()

            loss = resnet50_graph()

            end_t = time.time()
            if b % print_interval == 0:
                l = loss.numpy()
                of_losses.append(l)
                print(
                    "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                        epoch, b, l, end_t - start_t
                    )
                )

        print("epoch %d train done, start validation" % epoch)

        resnet50_module.eval()
        correct_of = 0.0
        for b in range(len(val_data_loader)):
            start_t = time.time()
            predictions, label = resnet50_eval_graph()
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()
            for i in range(args.val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()

        top1 = correct_of / all_samples
        of_accuracy.append(top1)
        print("epoch %d, oneflow top1 val acc: %f" % (epoch, top1))

        flow.save(
            resnet50_module.state_dict(),
            os.path.join(
                args.save_checkpoint_path,
                "epoch_%d_val_acc_%f" % (epoch, correct_of / all_samples),
            ),
        )

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
