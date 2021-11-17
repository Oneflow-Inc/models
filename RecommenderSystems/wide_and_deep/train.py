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
from models.dataloader_utils import make_data_loader
from oneflow.framework import distribute
from models.wide_and_deep import make_wide_and_deep_module
from oneflow.nn.parallel import DistributedDataParallel as ddp
from graph import WideAndDeepGraph, WideAndDeepTrainGraph
import warnings
import pandas as pd


class Trainer(object):
    def __init__(self):
        args = get_args()
        self.args = args
        self.execution_mode = args.execution_mode
        self.ddp = args.ddp
        if self.ddp == 1 and self.execution_mode == "graph":
            warnings.warn(
                """when ddp is True, the execution_mode can only be eager, but it is graph""",
                UserWarning,
            )
            self.execution_mode = "eager"
        self.is_consistent = (
            flow.env.get_world_size() > 1 and not args.ddp
        ) or args.execution_mode == "graph"
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        (
            self.train_dataloader,
            self.val_dataloader,
            self.wdl_module,
            self.loss,
            self.opt,
        ) = self.prepare_modules()
        if self.execution_mode == "graph":
            self.eval_graph = WideAndDeepGraph(
                self.wdl_module, self.val_dataloader, self.loss
            )
            self.train_graph = WideAndDeepTrainGraph(
                self.wdl_module, self.train_dataloader, self.loss, self.opt
            )
        self.record=[]

    def prepare_modules(self):
        args = self.args
        is_consistent = self.is_consistent
        self.wdl_module = make_wide_and_deep_module(args)
        if is_consistent == True:
            world_size = self.world_size
            placement = flow.placement("cuda", {0: range(world_size)})
            self.wdl_module = self.wdl_module.to_consistent(
                placement=placement, sbp=flow.sbp.broadcast
            )
        else:
            self.wdl_module=self.wdl_module.to("cuda")
        if args.model_load_dir != "":
            self.load_state_dict()
        if self.ddp:
            self.wdl_module = ddp(self.wdl_module)
        if args.save_initial_model and args.model_save_dir != "":
            self.save(os.path.join(args.model_save_dir, "initial_checkpoint"))

        train_dataloader = make_data_loader(args, mode="train", is_consistent=self.is_consistent)
        val_dataloader = make_data_loader(args, mode="val", is_consistent=self.is_consistent)

        bce_loss = flow.nn.BCELoss(reduction="none")
        bce_loss.to("cuda")

        opt = flow.optim.SGD(
            self.wdl_module.parameters(), lr=args.learning_rate, momentum=0.9
        )

        return train_dataloader, val_dataloader, self.wdl_module, bce_loss, opt

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
        if save_path is None:
            return
        print(f"Saving model to {save_path}")
        state_dict = self.wdl_module.state_dict()
        if self.is_consistent:
            flow.save(state_dict, save_path, consistent_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return

    def print_eval_metrics(self, step, loss, lables_list, predicts_list):
        all_labels = np.concatenate(lables_list, axis=0)
        all_predicts = np.concatenate(predicts_list, axis=0)
        auc = (
            "NaN"
            if np.isnan(all_predicts).any()
            else roc_auc_score(all_labels, all_predicts)
        )
        print(f"device {self.rank}: iter {step} eval_loss {loss} auc {auc}")

    def __call__(self):
        self.train()

    def train(self):
        losses = []
        args = self.args
        for i in range(args.max_iter):
            loss = self.train_one_step()
            record_loss = tol(loss, False)
            if self.ddp:
                # In ddp mode, the loss needs to be averaged
                record_loss = flow.comm.all_reduce(record_loss)
                losses.append(record_loss.numpy() / self.world_size)
            else:
                losses.append(record_loss.numpy())

            if (i + 1) % args.print_interval == 0 and self.rank == 0:
                l = sum(losses) / len(losses)
                losses = []
                if (i + 1) % 100 == 0:
                    print(f"iter {i+1} train_loss {l} time {time.time()}")
                self.to_record(i+1, l)
                if args.val_batch_size <= 0:
                    continue
                eval_loss_acc = 0.0
                lables_list = []
                predicts_list = []
                for j in range(args.val_batch_size):
                    predicts, labels, eval_loss = self.eval_one_step()
                    eval_loss_acc += eval_loss.numpy()
                    lables_list.append(labels.numpy())
                    predicts_list.append(predicts.numpy())
                self.print_eval_metrics(
                    i + 1, eval_loss_acc / args.val_batch_size, lables_list, predicts_list
                )
                self.wdl_module.train()
            time_begin=time.time()
        if self.rank == 0: 
            self.record_to_csv()

    def eval_one_step(self):
        self.wdl_module.eval()
        if self.execution_mode == "graph":
            predicts, labels, eval_loss = self.eval_graph()
        else:
            predicts,labels,eval_loss=self.forward()
        return predicts, labels, eval_loss

    def forward(self):
        (
            labels,
            dense_fields,
            wide_sparse_fields,
            deep_sparse_fields,
        ) = self.train_dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")
        predicts = self.wdl_module(
            dense_fields, wide_sparse_fields, deep_sparse_fields
        )
        loss = self.loss(predicts,labels)
        reduce_loss = flow.mean(loss)
        return predicts,labels,reduce_loss

    def train_eager(self):
        predicts,labels,loss = self.forward()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return predicts,labels,loss

    def train_one_step(self):
        self.wdl_module.train()
        if self.execution_mode == "graph":
            predicts, labels, train_loss = self.train_graph()
        else:
            predicts, labels, train_loss = self.train_eager()
        return train_loss


    def get_memory_usage(self):
        currentPath=os.path.dirname(os.path.abspath(__file__))
        nvidia_smi_report_file_dir = os.path.join(currentPath, 'log/gpu_info')
        isExists=os.path.exists(nvidia_smi_report_file_dir)
        if not isExists:
             os.makedirs(nvidia_smi_report_file_dir)
        nvidia_smi_report_file_path = os.path.join(nvidia_smi_report_file_dir, 'gpu_memory_usage_%s.csv' % self.rank)
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv"
        if nvidia_smi_report_file_path is not None:
            cmd += f" -f {nvidia_smi_report_file_path}"
        os.system(cmd)
        df=pd.read_csv(nvidia_smi_report_file_path)
        memory=df.iat[self.rank,1].split()[0]
        return memory
  
    
    def record_to_csv(self):
        currentPath=os.path.dirname(os.path.abspath(__file__))
        dir_path=os.path.join(currentPath, 'log/%s' % (self.args.test_name))
        isExists=os.path.exists(dir_path)
        if not isExists:
             os.makedirs(dir_path) 
        filePath=os.path.join(dir_path,'record_%s_%s.csv'%(self.args.batch_size, self.rank))
        df_record=pd.DataFrame.from_dict(self.record, orient='columns')
        df_record.to_csv(filePath,index=False)
        print("Record info is writting to %s" % filePath)
    
    def to_record(self,iter=0,loss=0,latency=0):
        data={}
        data['node']=1
        data['device']=self.rank
        data['batch_size']=self.args.batch_size
        data['deep_vocab_size']=self.args.deep_vocab_size
        data['deep_embedding_vec_size']=self.args.deep_embedding_vec_size
        data['hidden_units_num']=self.args.hidden_units_num
        data['iter']=iter
        data['memory_usage/MB']=self.get_memory_usage()
        data['loss']=loss      
        self.record.append(data)


def tol(tensor, pure_local=True):
    """ to local """
    if tensor.is_consistent:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local()

    return tensor


if __name__ == "__main__":
    trainer = Trainer()
    trainer()
