import argparse
import numpy as np
import os
import time
import datetime
import shutil
from tqdm import tqdm
import oneflow as flow
import torch

from model.oneflow_alexnet import alexnet
from utils.ofrecord_data_utils import OFRecordDataLoader

def _parse_args():
    parser = argparse.ArgumentParser("flags for test pretrained alexnet weight")
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--ofrecord_path", type=str, default="./ofrecord", help="dataset path"
    )
    parser.add_argument("--val_batch_size", type=int, default=512, help="val batch size")
    parser.add_argument("--weight_style", type=str, default="pytorch", choices=['pytorch', 'oneflow'], help="pretrained weight type")

    return parser.parse_args()

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def valid(args, model, criterion, data_loader):
    # Validation
    eval_losses = AverageMeter()

    model.eval()
    all_preds, all_label = [], []

    for steps in tqdm(range(len(data_loader))):
        # get data
        image, label = data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        with flow.no_grad():
            logits = model(image)
            preds = logits.softmax()
            eval_loss = criterion(logits, label)
            eval_losses.update(eval_loss.numpy())

        preds = preds.numpy()
        preds = np.argmax(preds, axis=-1)
        label = label.numpy()

        # collect results
        if len(all_preds) == 0:
            all_preds.append(preds)
            all_label.append(label)
        else:
            all_preds[0] = np.append(all_preds[0], preds, axis=0)
            all_label[0] = np.append(all_label[0], label, axis=0)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    print("Validation Results")
    print("Valid Loss: %2.5f" % eval_losses.avg)
    print("Valid Accuracy: %2.5f" % accuracy)
    return accuracy

def load_from_pytorch(load_checkpoint, model):
    print("Loading Pytorch Pretrained Weight Now")
    parameters = torch.load(load_checkpoint)
    new_parameters = dict()
    for key,value in parameters.items():
        if "num_batches_tracked" not in key:
            val = value.detach().cpu().numpy()
            new_parameters[key] = val
    model.load_state_dict(new_parameters)
    return model

def load_from_oneflow(load_checkpoint, model):
    print("Loading OneFlow Pretrained Weight Now")
    model.load_state_dict(flow.load(load_checkpoint))
    return model


def main(args):
    # Check Hyper-parameter
    assert args.load_checkpoint, "You should set load_checkpoint args for loading pretrained weights"

    # Dataset Setup
    val_data_loader = OFRecordDataLoader(
    ofrecord_root=args.ofrecord_path,
    mode="validation",
    dataset_size=50000,
    batch_size=args.val_batch_size,
    )

    # Model Init
    print("***** Initialization *****")
    start_t = time.time()
    model = alexnet()
    if args.load_checkpoint != "":
        print("load_checkpoint >>>>>>>>> ", args.load_checkpoint)
        if args.weight_style == "pytorch":
            model = load_from_pytorch(args.load_checkpoint, model)
        elif args.weight_style == "oneflow":
            model = load_from_oneflow(args.load_checkpoint, model)
        else:
            print("Only support oneflow type and pytorch type pretrained weight now")
            return
    end_t = time.time()
    print("init time : {}".format(end_t - start_t))

    # Testing Hyper-parameters
    criterion = flow.nn.CrossEntropyLoss()
    model.to("cuda")
    criterion.to("cuda")

    # Run Validation
    accuracy = valid(args, model, criterion, val_data_loader)
    print("Finished!")
    return 

if __name__ == "__main__":
    args = _parse_args()
    main(args)