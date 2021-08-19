import argparse
import logging
import os

import oneflow as flow
import oneflow.nn as nn

import losses
from backbones import get_model

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import MXFaceDataset, DataLoaderX

from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import numpy as np





class FC7(nn.Module):
    def __init__(self,input_size,output_size,bias=False ):
        super(FC7, self).__init__()
        self.fc7=nn.Linear(input_size,output_size,bias)
    
    def forward(self, x):
        return self.fc7(x) 
      




def main(args):
    cfg = get_config(args.config)
    world_size=1
    rank = 0
    local_rank = args.local_rank
  
    
    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root,rank, cfg.output)


    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to("cuda")
    fc7=FC7(cfg.embedding_size,cfg.num_classes).to("cuda")



    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(flow.load(backbone_pth))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")


    
    margin_softmax = losses.get_loss(cfg.loss)





    opt_backbone = flow.optim.SGD(backbone.parameters(),
        lr=cfg.lr,momentum=0.9, weight_decay=cfg.weight_decay)
    opt_fc7 = flow.optim.SGD(fc7.parameters(),
        lr=cfg.lr,  momentum=0.9, weight_decay=cfg.weight_decay)

    
    train_data_path="/home/zhuwang/ms1m-retinaface-t1"
    train_set = MXFaceDataset(root_dir=train_data_path, local_rank=0)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size , shuffle=True, num_workers=12,drop_last=True)

    num_image = len(train_set)
    total_batch_size = cfg.batch_size 
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    def lr_step_func(current_step):
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])

    # scheduler_backbone = flow.optimizer.PiecewiseScalingScheduler(
    #     optimizer=opt_backbone, lr_lambda=lr_step_func)
    # scheduler_pfc = flow.optimizer.PiecewiseScalingScheduler(
    #     optimizer=opt_fc7, lr_lambda=lr_step_func)


    scheduler_backbone =flow.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=lr_step_func)
    scheduler_pfc = flow.optim.lr_scheduler.LambdaLR(
        optimizer=opt_fc7, lr_lambda=lr_step_func)


   # sparse_softmax = flow.F.sparse_softmax_cross_entropy()

    #sparse_softmax = flow.nn.CrossEntropyLoss()
    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    val_target = cfg.val_targets
    callback_verification = CallBackVerification(2000, rank, val_target, cfg.rec)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    start_epoch = 0
    global_step = 0
    #grad_amp = MaxClipGradScaler(cfg.batch_size, 128 * cfg.batch_size, growth_interval=100) if cfg.fp16 else None
    of_cross_entropy = flow.nn.CrossEntropyLoss()
    for epoch in range(start_epoch, cfg.num_epoch):
        backbone.train()
        fc7.train()

        for step, (img, label) in enumerate(train_loader):
            global_step += 1

            img=np.ascontiguousarray(img.permute(0,3,1,2).numpy(), 'float32')
            label=label.numpy().astype(int)
            img=flow.Tensor(img).to("cuda")
            label=flow.Tensor(label,dtype=flow.int32).to("cuda")
            features = backbone(img)
            #features=fc7(features)
            # features_fc1=flow.math.l2_normalize(input=features , axis=1, epsilon=1e-10)
            features_fc7=margin_softmax(features,label)
            loss=of_cross_entropy(features_fc7,label)


            loss.backward()
            opt_backbone.step()
            #opt_fc7.step()
            opt_backbone.zero_grad()
            #opt_fc7.zero_grad()
            callback_logging(global_step, loss.numpy(), epoch,False, scheduler_backbone.get_last_lr()[0])
            callback_verification(global_step, backbone)
            scheduler_backbone.step()
            scheduler_pfc.step()
        callback_checkpoint(global_step, backbone, fc7)



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='OneFlow ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
