import os
import torch
import oneflow as flow
import numpy as np
from tqdm import tqdm
from model.wide_resnet import wide_resnet50_2, wide_resnet101_2
from torch_loader import *

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

def convert_checkpoint(state_dict):
    new_parameters = dict()
    for key, value in state_dict.items():
        if "num_batches_tracked" not in key:
            val = value.detach().cpu().numpy()
            new_parameters[key] = val
    return new_parameters

# alexnet_module = alexnet()
# alexnet_module.load_state_dict(new_parameters)
# flow.save(alexnet_module.state_dict(), "alexnet_oneflow_model")

def main(model, checkpoint_path):


    # device
    device, device_ids = setup_device(1)

    # load checkpoint
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        new_parameters = dict()
        for key, value in state_dict.items():
            if "num_batches_tracked" not in key:
                val = value.detach().cpu().numpy()
                new_parameters[key] = val.astype(np.float32)
        # flow_state_dict = convert_checkpoint(state_dict)
        model.load_state_dict(new_parameters)
        print("Load pretrained weights from {}".format(checkpoint_path))

    # send model to device
    model = model.to("cuda")

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
    with flow.no_grad():
        pbar = tqdm(enumerate(data_loader), total=total_batch)
        for batch_idx, (data, target) in pbar:
            pbar.set_description("Batch {:05d}/{:05d}".format(batch_idx, total_batch))

            # data = data.to(device)
            # target = target.to(device)
            data = flow.tensor(data.numpy()).to("cuda")
            target = target.to("cuda")
            # target = flow.tensor(target.numpy()).to("cuda")

            pred_logits = model(data)
            pred_logits = torch.tensor(pred_logits.numpy()).to(device)
            acc1, acc5 = accuracy(pred_logits, target, topk=(1, 5))

            acc1s.append(acc1.item())
            acc5s.append(acc5.item())

            pbar.set_postfix(acc1=acc1.item(), acc5=acc5.item())

    print("Evaluation of model {:s} on dataset {:s}, Acc@1: {:.4f}, Acc@5: {:.4f}".format("wide_resnet50_2", "ImageNet", np.mean(acc1s), np.mean(acc5s)))

if __name__ == "__main__":
    wide_resnet50_2_weight_path = "/data/rentianhe/code/new_models/models/wide_resnet/weight/torch/wide_resnet50_2-95faca4d.pth"
    wide_resnet101_2_weight_path = "/data/rentianhe/code/new_models/models/wide_resnet/weight/torch/wide_resnet101_2-32ee1156.pth"
    model = wide_resnet50_2()
    main(model, wide_resnet50_2_weight_path)