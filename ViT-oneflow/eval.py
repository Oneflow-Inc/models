import os
import torch
import oneflow as flow
import numpy as np
from tqdm import tqdm
from model import VisionTransformer
from config import get_eval_config
from torch_loader import *


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


def main():

    config = get_eval_config()

    # create model
    model = VisionTransformer(
             image_size=(config.image_size, config.image_size),
             patch_size=(config.patch_size, config.patch_size),
             emb_dim=config.emb_dim,
             mlp_dim=config.mlp_dim,
             num_heads=config.num_heads,
             num_layers=config.num_layers,
             num_classes=config.num_classes,
             attn_dropout_rate=config.attn_dropout_rate,
             dropout_rate=config.dropout_rate)

    # load checkpoint
    if config.checkpoint_path:
        state_dict = flow.load(config.checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))


    # create dataloader
    data_loader = eval("{}DataLoader".format(config.dataset))(
                    # data_dir=os.path.join(config.data_dir, config.dataset),
                    data_dir = config.data_dir,
                    image_size=config.image_size,
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                    split='val')
    total_batch = len(data_loader)

    # starting evaluation
    print("Starting evaluation")
    acc1s = []
    acc5s = []
    model.to("cuda")
    model.eval()
    with flow.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))
            # convert pytorch tensor to flow tensor
            data = flow.tensor(data.numpy())

            # load oneflow model and process data
            data = data.to("cuda")
            pred_logits = model(data)

            # convert to pytorch and evaluation
            pred_logits = torch.tensor(pred_logits.numpy())
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format(config.model_arch, config.dataset, np.mean(acc1s), np.mean(acc5s)))


if __name__ == '__main__':
    main()