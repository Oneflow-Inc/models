import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
from tqdm import tqdm
from config import get_args
from models.dataloader_utils import OFRecordDataLoader
from oneflow.framework import distribute
from models.wide_and_deep_module import WideAndDeep
from util import dump_to_npy, save_param_npy
from oneflow.nn.parallel import DistributedDataParallel as ddp
from graph import WideAndDeepGraph,WideAndDeepTrainGraph
import pandas as pd
from datetime import datetime



class Trainer(object):
    def __init__(self):
        args = get_args()
        self.args=args
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.execution_mode=args.execution_mode
        self.ddp=args.ddp
        self.is_consistent= (flow.env.get_world_size() > 1 and not args.ddp) or args.execution_mode=='graph'
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.train_dataloader, self.val_dataloader, self.wdl_module, self.loss, self.opt = self.prepare_modules()
        if self.execution_mode=='graph':
            self.eval_graph = WideAndDeepGraph(self.wdl_module,self.val_dataloader,self.loss)
            self.train_graph = WideAndDeepTrainGraph(self.wdl_module,self.train_dataloader,self.loss,self.opt)   
        self.record=[]

    def get_memory_usage(self):
        nvidia_smi_report_file_='gpu_memory_usage_%s.csv'%self.rank
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
        if nvidia_smi_report_file_ is not None:
            cmd += f" -f {nvidia_smi_report_file_}"
        os.system(cmd)
        currentPath=os.getcwd()
        filePath=os.path.join(currentPath,nvidia_smi_report_file_)
        df=pd.read_csv(filePath)
        memory=df.iat[self.rank,1].split()[0]
        return memory
        
    def record_to_csv(self):
        currentPath=os.getcwd()
        filePath=os.path.join(currentPath,'csv/%s_record_%s.csv'%(self.eval_name,self.rank))
        df_record=pd.DataFrame.from_dict(self.record, orient='columns')
        df_record.to_csv(filePath,index=False)

    def to_record(self,iter=0,loss=0,latency=0):
        data={}
        data['node']=1
        data['device']=self.rank
        data['batch_size']=self.args.batch_size
        data['deep_vocab_size']=self.args.deep_vocab_size
        data['deep_embedding_vec_size']=self.args.deep_embedding_vec_size
        data['hidden_units_num']=self.args.hidden_units_num
        data['iter']=iter
        data['latency/ms']=latency
        data['memory_usage/MB']=self.get_memory_usage()
        data['loss']=loss      
        self.record.append(data)

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
        losses = []
        args=self.args
        latency=0
        time_begin=time.time()
        for i in range(args.max_iter):
            loss=self.train_one_step()
            losses.append(loss)
            if (i+1) % args.print_interval == 0:
                time_end=time.time()
                latency=(time_end-time_begin)*1000/args.print_interval
                print(latency)
                l = sum(losses) / len(losses)
                self.to_record(i+1,l,round(latency,3))
                losses = []
                latency=0
                time_begin=time.time()            
        self.record_to_csv()

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
        if self.is_consistent==True and eval_loss.sbp==flow.sbp.partial_sum:
            eval_loss = eval_loss / self.world_size
        return predicts, labels, eval_loss

    def train_one_step(self):
        self.wdl_module.train()
        if self.execution_mode=='graph':
            predicts, labels, train_loss = self.train_graph()
        else:
            labels, dense_fields, wide_sparse_fields, deep_sparse_fields = self.train_dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields)
            train_loss = self.loss(predicts, labels)
        if self.is_consistent==True and train_loss.sbp==flow.sbp.partial_sum:
            train_loss = train_loss / self.world_size
        if self.is_consistent==True:
            return train_loss.to_local().numpy().mean()
        else:   
            train_loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            return train_loss.numpy().mean()



if __name__ == "__main__":
    trainer = Trainer()
    trainer()
