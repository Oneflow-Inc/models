import argparse
import numpy as np
import os
import time
import oneflow as flow

from model.vit import *
from utils.ofrecord_data_utils import OFRecordDataLoader
from model.build_model import build_model


def _parse_args():
    parser = argparse.ArgumentParser("flags for train mobilenet_v2")
    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        default="./checkpoints",
        help="save checkpoint root dir",
    )
    parser.add_argument(
        "--model_arch",
        type=str,
        default="vit_b_16_384",
        choices=[
            "vit_b_16_224",
            "vit_b_16_384",
            "vit_b_32_224",
            "vit_b_32_384",
            "vit_l_16_224",
            "vit_l_16_384",
        ],
        help="model architecture",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint",
    )
    parser.add_argument(
        "--image_size", type=int, default=384, help="default train image size"
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
        image_size=args.image_size,
    )

    val_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        mode="val",
        dataset_size=3925,
        batch_size=args.val_batch_size,
        image_size=args.image_size,
    )

    # oneflow init
    start_t = time.time()
    model = build_model(args)
    if args.load_checkpoint != "":
        print("load_checkpoint >>>>>>>>> ", args.load_checkpoint)
        checkpoint = flow.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    of_cross_entropy = flow.nn.CrossEntropyLoss()

    model.to("cuda")
    of_cross_entropy.to("cuda")

    of_sgd = flow.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.mom
    )

    of_losses = []
    all_samples = len(val_data_loader) * args.val_batch_size
    print_interval = 20

    for epoch in range(args.epochs):
        model.train()

        for b in range(len(train_data_loader)):
            image, label = train_data_loader()

            # oneflow train
            start_t = time.time()
            image = image.to("cuda")
            label = label.to("cuda")
            logits = model(image)
            loss = of_cross_entropy(logits, label)
            loss.backward()
            of_sgd.step()
            of_sgd.zero_grad()
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

        model.eval()
        correct_of = 0.0
        for b in range(len(val_data_loader)):
            image, label = val_data_loader()

            start_t = time.time()
            image = image.to("cuda")
            with flow.no_grad():
                logits = model(image)
                predictions = logits.softmax()
            of_predictions = predictions.numpy()
            clsidxs = np.argmax(of_predictions, axis=1)

            label_nd = label.numpy()
            for i in range(args.val_batch_size):
                if clsidxs[i] == label_nd[i]:
                    correct_of += 1
            end_t = time.time()

        print("epoch %d, oneflow top1 val acc: %f" % (epoch, correct_of / all_samples))

        flow.save(
            model.state_dict(),
            os.path.join(
                args.save_checkpoint_path,
                "epoch_%d_val_acc_%f" % (epoch, correct_of / all_samples),
            ),
        )

    writer = open("of_losses.txt", "w")
    for o in of_losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
