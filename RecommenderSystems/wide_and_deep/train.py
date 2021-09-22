import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
from config import get_args
from models.dataloader_utils import OFRecordDataLoader
from oneflow.framework import distribute
from models.wide_and_deep import WideAndDeep
from oneflow.nn.parallel import DistributedDataParallel as ddp
from graph import WideAndDeepGraph,WideAndDeepTrainGraph
import warnings


class Trainer(object):
    
    def __init__(self):
        args = get_args()
        self.args=args
        self.execution_mode=args.execution_mode
        self.ddp=args.ddp
        if self.ddp==1 and self.execution_mode=='graph':
            warnings.warn('''when ddp is True, the execution_mode can only be eager, but it is graph''', UserWarning)
            self.execution_mode='eager'
        self.is_consistent= (flow.env.get_world_size() > 1 and not args.ddp) or args.execution_mode=='graph'
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.train_dataloader, self.val_dataloader, self.wdl_module, self.loss, self.opt = self.prepare_modules()
        if self.execution_mode=='graph':
            self.eval_graph = WideAndDeepGraph(self.wdl_module,self.val_dataloader,self.loss)
            self.train_graph = WideAndDeepTrainGraph(self.wdl_module,self.train_dataloader,self.loss,self.opt)   
        
    def prepare_modules(self):
        args=self.args
        is_consistent=self.is_consistent
        world_size = self.world_size
        placement = flow.placement("cpu", {0: range(world_size)})
        wdl_module = WideAndDeep(args)
        if is_consistent==True:
            wdl_module = wdl_module.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        train_dataloader = OFRecordDataLoader(args)
        val_dataloader = OFRecordDataLoader(args, mode="val")
        wdl_module.to("cuda")
        bce_loss = flow.nn.BCELoss(reduction="mean")
        bce_loss.to("cuda")
        opt = flow.optim.SGD(wdl_module.parameters(), lr=args.learning_rate, momentum=0.9)
        if args.model_load_dir != "":
            self.load_state_dict()
        if args.save_initial_model and args.model_save_dir != "":
            self.save(os.path.join(args.model_save_dir, "initial_checkpoint"))
        return train_dataloader, val_dataloader, wdl_module, bce_loss, opt

    def load_state_dict(self):
        print(f"Loading model from {self.args.model_load_dir}")
        if self.is_consistent:
            state_dict = flow.load(self.args.model_load_dir, consistent_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.args.model_load_dir)
        else:
            return
        self.wdl_module.load_state_dict(state_dict)

    def save(self, save_path):
        if self.save_path is None:
            return
        print(f"Saving model to {save_path}")
        state_dict = self.wdl_module.state_dict()
        if self.is_consistent:
            flow.save(state_dict, save_path, consistent_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return

    def print_eval_metrics(self,step, loss, lables_list, predicts_list):
        all_labels = np.concatenate(lables_list, axis=0)
        all_predicts = np.concatenate(predicts_list, axis=0)
        auc = (
            "NaN"
            if np.isnan(all_predicts).any()
            else roc_auc_score(all_labels, all_predicts)
        )
        rank=flow.env.get_rank()
        print(f"device {rank}: iter {step} eval_loss {loss} auc {auc}")

    def __call__(self):
        self.train()

    def train(self):
        def handle(dict):
            for key,value in dict.items():
                if self.is_consistent==True:
                    if key=='loss':
                        print('device:%s %s \n'%(flow.env.get_rank(),dict[key]))
                    dict[key]=value.to_consistent(placement=flow.placement("cuda", {0: range(self.world_size)}), sbp=flow.sbp.broadcast).to_local().numpy()
                    if key=='loss':
                        print(dict[key])
                else:   
                    dict[key]=value.numpy()
            return dict
        losses = []
        args=self.args
        for i in range(args.max_iter):
            loss=self.train_one_step()
            losses.append(handle({'loss':loss})['loss'])
            if self.execution_mode=='eager':
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            if (i+1) % args.print_interval == 0:
                l = sum(losses) / len(losses)
                losses = []
                rank=flow.env.get_rank()
                print(f"device {rank}: iter {i+1} train_loss {l} time {time.time()}")
                if args.eval_batchs <= 0:
                    continue
                eval_loss_acc = 0.0
                lables_list = []
                predicts_list = []               
                for j in range(args.eval_batchs):
                    predicts, labels, eval_loss=self.eval_one_step()
                    dict=handle({
                        'predicts':predicts,
                        'labels':labels,
                        'eval_loss':eval_loss
                    })
                    eval_loss_acc += dict['eval_loss']
                    lables_list.append(dict['labels'])
                    predicts_list.append(dict['predicts'])
                self.print_eval_metrics(i+1, eval_loss_acc/args.eval_batchs,
                                lables_list, predicts_list)
                self.wdl_module.train()

    def eval_one_step(self):
        self.wdl_module.eval()
        if self.execution_mode=='graph':
            predicts, labels, eval_loss = self.eval_graph()
        else:
            labels, dense_fields, wide_sparse_fields, deep_sparse_fields = self.val_dataloader()
            labels = labels.to(dtype=flow.float32).to("cuda")
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields)
            eval_loss = self.loss(predicts, labels)
        return predicts, labels, eval_loss

    def train_one_step(self):
        self.wdl_module.train()
        if self.execution_mode=='graph':
            predicts, labels, train_loss = self.train_graph()
            print('predicts.sbp',predicts.sbp)
            print('labels.sbp',labels.sbp)
            print('train_loss.sbp',train_loss.sbp)
        else:
            labels, dense_fields, wide_sparse_fields, deep_sparse_fields = self.train_dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields)
            train_loss = self.loss(predicts, labels)
        return train_loss

if __name__ == "__main__":
    trainer = Trainer()
    trainer()
