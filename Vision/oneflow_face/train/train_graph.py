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




def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        combine_margin,
        cross_entropy,
        data_loader,
        optimizer,
        lr_scheduler=None,
        return_pred_and_label=True,
    ):
        super().__init__()
        #args = get_args()
        self.return_pred_and_label = return_pred_and_label

        # if args.use_fp16:
        #     self.config.enable_amp(True)
        #     self.set_grad_scaler(make_grad_scaler())
        # elif args.scale_grad:
        self.set_grad_scaler(make_static_grad_scaler())

        # self.config.allow_fuse_add_to_output(True)
        # self.config.allow_fuse_model_update_ops(True)

        self.model = model

        self.cross_entropy = cross_entropy
        self.combine_margin=combine_margin
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        image, label = self.data_loader()

        image = image.to("cuda")
        label = label.to("cuda")

        logits = self.model(image)
        logits =self.combine_margin(logits,label)*64
        loss = self.cross_entropy(logits, label)

        loss.backward()
        return loss




class FC7(flow.nn.Module):
    def __init__(self,input_size,output_size,bias=False ):
        super(FC7, self).__init__()

        self.weight = flow.nn.Parameter(flow.empty(input_size,output_size))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)
 
          
    def forward(self, x):
       
        x=flow.nn.functional.l2_normalize(input=x , dim=1, epsilon=1e-10)
        weight=flow.nn.functional.l2_normalize(input=self.weight , dim=0, epsilon=1e-10)
              
        x=flow.matmul(x,weight)
        return x


class Train_Module(flow.nn.Module):
    def __init__(self,cfg,backbone,placement,world_size,bias=False ):
        super(Train_Module, self).__init__()
        self.placement=placement
        if cfg.model_parallel:
            input_size=cfg.embedding_size
            output_size=int(cfg.num_classes/world_size)
            self.fc = FC7(input_size,output_size,bias=bias).to_consistent(placement=placement, sbp = flow.sbp.split(1))
        else:
            self.fc = FC7(cfg.embedding_size,cfg.num_classes,bias=bias).to_consistent(placement=placement, sbp = flow.sbp.broadcast)
        self.backbone=backbone.to_consistent(placement=placement, sbp = flow.sbp.broadcast)
    
    def forward(self, x):
        x=self.backbone(x)
        if x.is_consistent:
            x = x.to_consistent(sbp=flow.sbp.broadcast)
        x=self.fc(x)
        return x



def make_optimizer(args, model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    # if args.grad_clipping > 0.0:
    #     assert args.grad_clipping == 1.0, "ONLY support grad_clipping == 1.0"
    #     param_group["clip_grad_max_norm"] = (1.0,)
    #     param_group["clip_grad_norm_type"] = (2.0,)

    optimizer = flow.optim.SGD(
        [param_group],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optimizer




class EvalGraph(flow.nn.Graph):
    def __init__(self, model):
        super().__init__()

        self.config.allow_fuse_add_to_output(True)

        self.model = model

    def build(self,image):

        image = image
        logits = self.model(image)
        return logits



def make_data_loader(args, mode, is_consistent=False, synthetic=False):
    assert mode in ("train", "validation")

    if mode == "train":
        total_batch_size = args.batch_size*flow.env.get_world_size()
        batch_size = args.batch_size
        num_samples = args.num_image
    else:
        total_batch_size = args.val_global_batch_size
        batch_size = args.val_batch_size
        num_samples = args.val_samples_per_epoch

    placement = None
    sbp = None

    if is_consistent:
        world_size = flow.env.get_world_size()
        placement = flow.placement("cpu", {0: range(world_size)})
        sbp = flow.sbp.split(0)
        #sbp = flow.sbp.broadcast

        # NOTE(zwx): consistent view, only consider logical batch size
        batch_size = total_batch_size


  
    ofrecord_data_loader = OFRecordDataLoader(
        ofrecord_root=args.ofrecord_path,
        
        dataset_size=num_samples,
        batch_size=batch_size,
        total_batch_size=total_batch_size,
        data_part_num=args.ofrecord_part_num,
        placement=placement,
        sbp=sbp,
    )
    return ofrecord_data_loader


def meter(self, mkey, *args):
    assert mkey in self.m
    self.m[mkey]["meter"].record(*args)


def main(args):
    cfg = get_config(args.config)

    local_rank = args.local_rank 
    rank = flow.env.get_rank()
    world_size = flow.env.get_world_size()


    os.makedirs(cfg.output, exist_ok=True)
    log_root = logging.getLogger()
    init_logging(log_root,rank, cfg.output)


    placement = flow.placement("cuda", {0: range(world_size)})      

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to("cuda")
    train_module=Train_Module(cfg,backbone,placement,world_size).to("cuda")
    


    if cfg.resume:
        backbone_pth = os.path.join("lazy_r50", "snapshot_new")
        train_module.load_state_dict(flow.load(backbone_pth))
        if rank == 0:
            logging.info("backbone resume successfully!")

    

    if cfg.loss=="cosface":
        margin_softmax = flow.nn.CombinedMarginLoss(1,0.,0.4).to("cuda")
    else:
        margin_softmax = flow.nn.CombinedMarginLoss(1,0.5,0.).to("cuda")
    of_cross_entropy=flow.nn.CrossEntropyLoss().to("cuda")


    opt_fc7=make_optimizer(cfg,train_module)
    
    train_data_loader =make_data_loader(cfg,'train',True)


    num_image = cfg.num_image
    total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = num_image // total_batch_size * cfg.warmup_epoch
    cfg.total_step = num_image // total_batch_size * cfg.num_epoch


    cfg.decay_step = [x * num_image // total_batch_size for x in cfg.decay_epoch]

    scheduler_pfc = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=opt_fc7, milestones= cfg.decay_step, gamma=0.1 
    )    
  

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))



    val_target = cfg.val_targets
    callback_logging = CallBackLogging(50, rank, cfg.total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    losses = AverageMeter()
    start_epoch = 0
    global_step = 0



    train_graph =TrainGraph(train_module,margin_softmax,of_cross_entropy,train_data_loader,opt_fc7,scheduler_pfc)
    train_graph.debug()
    val_graph =EvalGraph(backbone)



    for epoch in range(start_epoch, cfg.num_epoch):
        train_module.train()

        for steps in range(len(train_data_loader)):    
            
            loss=train_graph()
            loss=loss.to_local().numpy()*world_size
            losses.update(loss, 1)
           
            callback_logging(global_step, losses, epoch,False, scheduler_pfc.get_last_lr()[0])
            global_step += 1
        callback_checkpoint(global_step, epoch, train_module,is_consistent=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='OneFlow ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
