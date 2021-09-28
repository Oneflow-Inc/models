import argparse
import logging
import os

import oneflow as flow
import oneflow.nn as nn


import sys
sys.path.append("..")
import losses
from backbones import get_model






from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
import numpy as np

import   pickle
import time
from utils.ofrecord_data_utils import OFRecordDataLoader
from oneflow.nn.parallel import DistributedDataParallel as ddp
      

class FC7(flow.nn.Module):
    def __init__(self,input_size,output_size,backbone,bias=False ):
        super(FC7, self).__init__()
        self.backbone=backbone
        #self.fc7=nn.Linear(input_size,output_size,bias)
        self.weight = flow.nn.Parameter(flow.empty(output_size,input_size))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)

        # size = args.device_num_per_node * args.num_nodes
        # num_local = (config.num_classes + size - 1) // size
        # num_sample = int(num_local * args.sample_ratio)
        # args.total_num_sample = num_sample * size


    def forward(self, x):
        x=self.backbone(x)             
        x=flow.nn.functional.l2_normalize(input=x , dim=1, epsilon=1e-10)
        weight=flow.nn.functional.l2_normalize(input=self.weight , dim=1, epsilon=1e-10)
        weight=weight.transpose(0,1)
        x=flow.matmul(x,weight)
        return x







def main(args):
    cfg = get_config(args.config)

    local_rank = args.local_rank 
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()


    # local_rank = args.local_rank 
    # rank = 0
    # world_size = 4


    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root,rank, cfg.output)


    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
    fc7=FC7(cfg.embedding_size,cfg.num_classes,backbone).to("cuda")
    fc7 = ddp(fc7)


    if cfg.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(flow.load(backbone_pth))
            if rank == 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            if rank == 0:
                logging.info("resume fail, backbone init successfully!")


    
    if cfg.loss=="cosface":
        margin_softmax = flow.nn.CombinedMarginLoss(1,0.,0.4).to("cuda")
    else:
        margin_softmax = flow.nn.CombinedMarginLoss(1,0.5,0.).to("cuda")
    of_cross_entropy=flow.nn.CrossEntropyLoss().to("cuda")


    opt_fc7 = flow.optim.SGD(fc7.parameters(),
        lr=cfg.lr,  momentum=0.9, weight_decay=cfg.weight_decay)

    


    train_data_loader = OFRecordDataLoader(
        ofrecord_root=cfg.ofrecord_path,
        mode="train",
        dataset_size=cfg.num_image,
        batch_size=cfg.batch_size,
        total_batch_size=cfg.batch_size* world_size,
        data_part_num=cfg.ofrecord_part_num
    )



    num_image = cfg.num_image
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch

    def lr_step_func(current_step):
        cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]
        if current_step < cfg.warmup_step:
            return current_step / cfg.warmup_step
        else:
            return 0.1 ** len([m for m in cfg.decay_step if m <= current_step])

    scheduler_pfc = flow.optim.lr_scheduler.LambdaLR(
        optimizer=opt_fc7, lr_lambda=lr_step_func)


    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    val_target = cfg.val_targets
    callback_verification = CallBackVerification(3000, rank, val_target, cfg.ofrecord_path,image_nums=cfg.val_image_num)
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    losses = AverageMeter()
    start_epoch = 0
    global_step = 0



    for epoch in range(start_epoch, cfg.num_epoch):
        fc7.train()
        #for step, (img, label) in enumerate(train_loader):
        for steps in range(len(train_data_loader)):    
            global_step += 1

            image, label = train_data_loader()
            image = image.to("cuda")
            label = label.to("cuda")
            features_fc7=fc7(image)
            features_fc7=margin_softmax(features_fc7,label)*64                    
            loss=of_cross_entropy(features_fc7,label)



            loss.backward()
            
            opt_fc7.step()
            opt_fc7.zero_grad()
            
            
            loss=loss.numpy()
            losses.update(loss, 1)
            callback_logging(global_step, losses, epoch,False, scheduler_pfc.get_last_lr()[0])
            callback_verification(global_step, backbone)
            scheduler_pfc.step()
        callback_checkpoint(global_step, epoch, fc7)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
