import os
import torch
from torch.utils.data import dataloader
import oneflow as flow
import numpy as np
from tqdm import tqdm
from model import VisionTransformer
from config import get_eval_config
from torch_loader import *
from ofrecord_data_utils import OFRecordDataLoader

# def build_loader(args):
#     train_data_loader = OFRecordDataLoader(
#         ofrecord_root=args.data_path,
#         mode="train",
#         dataset_size=args.num_train_examples,
#         batch_size=args.train_batch_size,
#     )

    # val_data_loader = OFRecordDataLoader(
    #     ofrecord_root=args.data_path,
    #     mode="validation",
    #     dataset_size=args.num_val_examples,
    #     batch_size=args.eval_batch_size,
    # )
#     return train_data_loader, val_data_loader

# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""""
#     """output size: (batch, 1000), target size: (batch, )"""
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].contiguous().view(-1).float().sum(0)
#         res.append(correct_k / batch_size * 100.0)
#     return res

def accuracy(output, target, topk=(1,)):
    """A numpy version of top-k accuracy"""
    batch_size = target.shape[0]
    output = output.numpy()
    target = target.numpy().reshape(batch_size, 1)
    res = []
    for k in topk:
        max_k_preds = output.argsort(axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == target, axis=1)
        topk_acc_score = (match_array.sum() / batch_size) * 100.0
        res.append(topk_acc_score)
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
    config.checkpoint_path = "./weight/ViT-B_16_oneflow"
    if config.checkpoint_path:
        state_dict = flow.load(config.checkpoint_path)
        model.load_state_dict(state_dict)
        print("Load pretrained weights from {}".format(config.checkpoint_path))



    data_loader = OFRecordDataLoader(
        ofrecord_root="/data/imagenet/ofrecord/",
        mode="validation",
        dataset_size=50000,
        batch_size=32,
    )

    total_batch = len(data_loader)

    # for step in range(len(val_data_loader)):
    #     image, label = val_data_loader.get_batch()
    #     print(image.shape)

    # starting evaluation
    print("Starting evaluation")
    acc1s = []
    acc5s = []
    model.to("cuda")
    model.eval()
    with flow.no_grad():
        for step in range(len(data_loader)):
            data, target = data_loader.get_batch()
            # load oneflow model and process data
            data = data.to("cuda")
            pred_logits = model(data)

            # convert to pytorch and evaluation
            # pred_logits = torch.tensor(pred_logits.numpy())
            # target = torch.tensor(target.numpy())
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())
            print("step: {:.4f}, acc1: {:.4f}, acc5: {:.4f}".format(step, acc1, acc5))

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format(config.model_arch, config.dataset, np.mean(acc1s), np.mean(acc5s)))


if __name__ == '__main__':
    main()