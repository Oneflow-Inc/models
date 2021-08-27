import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from model.build_model import build_model
from torch_loader import *


def _parse_args():
    parser = argparse.ArgumentParser("flags for eval acc")
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="torch",
        help="eval torch or flow model",
    )
    parser.add_argument(
        "--model", type=str, default="alexnet", help="choose the model to eval"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="the path to load pytorch pretrained weight"
    )
    return parser.parse_args()


def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res

def main(args):

    model = build_model(args)
    checkpoint_path = args.checkpoint_path
    # device
    device, device_ids = setup_device(1)

    # load checkpoint
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(checkpoint_path))

    # send model to device
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # create dataloader
    data_loader = eval("{}DataLoader".format("ImageNet"))(
                    data_dir="/data/imagenet/extract",
                    image_size=224,
                    batch_size=32,
                    num_workers=8,
                    split='val')
    total_batch = len(data_loader)

    # starting evaluation
    print("Starting evaluation")
    acc1s = []
    acc5s = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            data = data.to(device)
            target = target.to(device)

            pred_logits = model(data)
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format("wide_resnet50_2", "ImageNet", np.mean(acc1s), np.mean(acc5s)))

if __name__ == "__main__":
    args = _parse_args()
    main(args)
