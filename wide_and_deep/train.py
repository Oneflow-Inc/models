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



class Trainer(object):
    
    def __init__(self):
        '''
        是否用对齐pytorch那个ddp, ddp, bool
        eager还是graph, execution_mode, str
        数据并行还是模型并行, 不知道叫啥, str
        模型并行的方案, deep_embedding_table_split_axis, int
        
        '''
        args = get_args()
        self.args=args
        for k, v in args.__dict__.items():
            setattr(self, k, v)
        self.execution_mode=args.execution_mode #'graph' or 'eager'
        self.ddp=args.ddp
        self.is_consistent= (flow.env.get_world_size() > 1 and not args.ddp) or args.execution_mode=='graph'

        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()

        self.train_dataloader, self.val_dataloader, self.wdl_module, self.loss, self.opt = self.prepare_modules()

        if self.execution_mode=='graph':
            self.eval_graph = WideAndDeepGraph(self.wdl_module,self.val_dataloader,self.loss)
            self.train_graph = WideAndDeepTrainGraph(self.wdl_module,self.train_dataloader,self.loss,self.opt)   
        

    def prepare_modules(self):
        args=self.args
        execution_mode=self.execution_mode
        ddp=self.ddp
        is_consistent=self.is_consistent

        world_size = self.world_size
        placement = flow.placement("cpu", {0: range(world_size)})
        wdl_module = WideAndDeep(args)
        #对标pytorch的ddp
        if ddp and is_consistent==False:
            placement = None
            dataloader_sbp = None
        elif execution_mode=='graph' and ddp==False and is_consistent==True:
            #graph_train_consistent, graph+consistent下的数据并行
            dataloader_sbp = flow.sbp.split(0)
            wdl_module = wdl_module.to_consistent(placement=placement, sbp=flow.sbp.broadcast)


        elif execution_mode=='eager' and ddp==False and is_consistent==True:
            #eager_train_consistent, eager+consistent下的数据并行
            dataloader_sbp = flow.sbp.split(0)
            wdl_module = wdl_module.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        #模型并行
        # elif execution_mode=='graph' and mode=='dmp' and is_consistent==True:
        #     #graph_train_consistent_dmp, graph+consistent下的模型并行
        #     sbp = flow.sbp.broadcast
        # elif execution_mode=='eager' and mode=='dmp' and is_consistent==True:
        #     #eager_train_consistent_dmp, eager+consistent下的模型并行
        #     sbp = flow.sbp.broadcast
        else:
            print('不支持的训练模式')
            return
        train_dataloader = OFRecordDataLoader(args, data_root=args.data_dir, batch_size=args.batch_size,placement=placement,sbp=dataloader_sbp)
        val_dataloader = OFRecordDataLoader(args, data_root=args.data_dir, mode="val",batch_size=args.batch_size,placement=placement,sbp=dataloader_sbp)

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
        for i in tqdm(range(args.max_iter)):
            loss=self.train_one_step()
            losses.append(loss)
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
                
                    #各个rank 打印local loss
                    if self.is_consistent==True:
                        eval_loss_acc += eval_loss.to_local().numpy().mean()
                        lables_list.append(labels.to_local().numpy())
                        predicts_list.append(predicts.to_local().numpy())
                    else:
                        eval_loss_acc += eval_loss.numpy().mean()
                        lables_list.append(labels.numpy())
                        predicts_list.append(predicts.numpy())
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
        #train_loss是partial_sum，比如consistent模式下的ddp
        if self.is_consistent==True and train_loss.sbp==flow.sbp.partial_sum:
            train_loss = train_loss / self.world_size

        #各个rank 打印local loss
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
