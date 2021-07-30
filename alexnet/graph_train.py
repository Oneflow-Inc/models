import oneflow as flow
import argparse
import numpy as np
import os
import time

from models.alexnet import alexnet
from utils.ofrecord_data_utils import OFRecordDataLoader


def _parse_args():
    parser = argparse.ArgumentParser("flags for train alexnet")
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

    return parser.parse_args()


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
    alexnet_module = alexnet()
    if args.load_checkpoint != "":
        print("load_checkpoint >>>>>>>>> ", args.load_checkpoint)
        alexnet_module.load_state_dict(flow.load(args.load_checkpoint))

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    of_cross_entropy = flow.nn.CrossEntropyLoss()

    alexnet_module.to("cuda")
    of_cross_entropy.to("cuda")

    of_sgd = flow.optim.SGD(
        alexnet_module.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    class AlexNetGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.alexnet = alexnet_module
            self.cross_entropy = of_cross_entropy
            self.add_optimizer("sgd", of_sgd)
        
        def build(self, image, label):
            logits = self.alexnet(image)
            loss = self.cross_entropy(logits, label)
            loss.backward()
            return loss

    alexnet_graph = AlexNetGraph()

    # class AlexNetEvalGraph(flow.nn.Graph):
    #     def __init__(self):
    #         super().__init__()
    #         self.alexnet = alexnet_module
    #     
    #     def build(self, image):
    #         with flow.no_grad():
    #             logits = self.alexnet(image)
    #             predictions = logits.softmax()
    #         return predictions

    # alexnet_eval_graph = AlexNetEvalGraph()

    of_losses = []
    all_samples = len(val_data_loader) * args.val_batch_size
    print_interval = 100


    for epoch in range(args.epochs):
        alexnet_module.train()

        for b in range(len(train_data_loader)):
            image, label = train_data_loader.get_batch()

            # oneflow graph train
            start_t = time.time()
            image = image.to("cuda")
            label = label.to("cuda")

            loss = alexnet_graph(image, label)

            end_t = time.time()
            if b % print_interval == 0:
                l = loss.numpy()[0]
                of_losses.append(l)
                print(
                    "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                        epoch, b, l, end_t - start_t
                    )
                )

        # print("epoch %d train done, start validation" % epoch)

        # alexnet_module.eval()
        # correct_of = 0.0
        # for b in range(len(val_data_loader)):
        #     image, label = val_data_loader.get_batch()

        #     start_t = time.time()
        #     image = image.to("cuda")
        #     predictions = alexnet_eval_graph(image)

        #     of_predictions = predictions.numpy()
        #     clsidxs = np.argmax(of_predictions, axis=1)

        #     label_nd = label.numpy()
        #     for i in range(args.val_batch_size):
        #         if clsidxs[i] == label_nd[i]:
        #             correct_of += 1
        #     end_t = time.time()

        # print("epoch %d, oneflow top1 val acc: %f" % (epoch, correct_of / all_samples))

        # flow.save(
        #     alexnet_module.state_dict(),
        #     os.path.join(
        #         args.save_checkpoint_path,
        #         "epoch_%d_val_acc_%f" % (epoch, correct_of / all_samples),
        #     ),
        # )

    writer = open("graph_of_losses.txt", "w")
    for o in of_losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
