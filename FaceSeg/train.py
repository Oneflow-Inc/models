import oneflow as flow
import time
import argparse

from utils.numpy_data_utils import face_seg
from models.LinkNet34 import LinkNet34

import oneflow.nn as nn
import sys

from albumentations import (
    HorizontalFlip,
    Compose,
    Rotate,
)


def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    parser.add_argument(
        "--model_path", type=str, default="./resnet50-19c8e357.pth", help="model path"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="/home/zj/face-segmentation/data/", help="dataset path"
    )
    parser.add_argument(
        "--save_model_name", type=str, default="linknet34_oneflow_training_model_50_test3", help="save path"
    )
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument("--batch_size", type=int, default=4, help="")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="")
    parser.add_argument("--mom", type=float, default=0.9, help="")
    return parser.parse_args()


class LossBinary:
    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = flow.Tensor(
                (targets.numpy() == 1)).to(flow.device('cuda'))
            jaccard_output = flow.sigmoid(outputs)
            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()
            loss -= self.jaccard_weight * \
                flow.log((intersection + eps) / (union - intersection + eps))
        return loss


def main(args):
    train_aug = Compose([
        HorizontalFlip(p=0.5),
        Rotate(15),
    ])

    train_data_loader = face_seg(
        args.dataset_path, args.batch_size, augmentation=train_aug)
    print(len(train_data_loader))

    #################
    # oneflow init
    start_t = time.time()
    linknet34_module = LinkNet34(
        pretrained=True, pretrained_model_path=args.model_path)
    end_t = time.time()

    linknet34_module.to(flow.device('cuda'))
    of_sgd = flow.optim.SGD(linknet34_module.parameters(),
                            lr=args.learning_rate, momentum=args.mom)
    cosine = flow.optim.lr_scheduler.CosineAnnealingLR(of_sgd, 2)

    ############################
    of_losses = []
    criterion = LossBinary(jaccard_weight=1)
    for epoch in range(args.epochs):
        linknet34_module.train()

        train_data_loader.shuffle_data()
        epoch_loss = 0
        for b in range(len(train_data_loader)):

            of_sgd.zero_grad()
            image_nd, label_nd = train_data_loader[b]
            # oneflow train
            start_t = time.time()
            image = flow.Tensor(image_nd)
            label = flow.Tensor(
                label_nd, dtype=flow.float32, requires_grad=False)
            image = image.to(flow.device('cuda'))
            label = label.to(flow.device('cuda'))
            logits = linknet34_module(image)
            mask_probs_flat = flow.reshape(logits, shape=[-1])
            true_masks_flat = flow.reshape(label, shape=[-1])
            loss = criterion(mask_probs_flat, true_masks_flat)
            epoch_loss += loss.numpy()
            loss.backward()
            of_sgd.step()
            cosine.step()
            end_t = time.time()
            l = loss.numpy()
            of_losses.append(l)
            sys.stdout.write(
                f'\rEpoch: {epoch} ---- Loss: {round(epoch_loss / (b + 1), 4)} ----- num: {b}')
            sys.stdout.flush()

        print("epoch %d done, start validation" % epoch)
    flow.save(linknet34_module.state_dict(), args.save_model_name)

    writer = open("of_losses.txt", "w")
    for o in of_losses:
        writer.write("%f\n" % o)
    writer.close()


if __name__ == "__main__":
    args = _parse_args()
    main(args)