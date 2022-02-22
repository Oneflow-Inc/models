import oneflow as flow
import os
import sys
import time
import pickle

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import numpy as np
from sklearn.metrics import roc_auc_score
from config import get_args
from models.data import make_data_loader
from utils.petastorm_dataloader import make_petastorm_dataloader
from models.dlrm import make_dlrm_module
from lr_scheduler import make_lr_scheduler
from oneflow.nn.parallel import DistributedDataParallel as DDP
from graph import DLRMTrainGraphWithDataloader, DLRMValGraph, DLRMTrainGraph, DLRMValGraphWithDataloader
import warnings
from utils.auc_calculater import calculate_auc_from_dir


class Trainer(object):
    def __init__(self):
        args = get_args()
        self.args = args
        self.save_path = args.model_save_dir
        self.save_init = args.save_initial_model
        self.save_model_after_each_eval = args.save_model_after_each_eval
        self.eval_after_training = args.eval_after_training
        self.dataset_format = args.dataset_format
        self.execution_mode = args.execution_mode
        self.max_iter = args.max_iter
        self.loss_print_every_n_iter = args.loss_print_every_n_iter
        self.ddp = args.ddp
        if self.ddp == 1 and self.execution_mode == "graph":
            warnings.warn(
                """when ddp is True, the execution_mode can only be eager, but it is graph""",
                UserWarning,
            )
            self.execution_mode = "eager"
        self.is_global = args.is_global
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.cur_iter = 0
        self.eval_interval = args.eval_interval
        self.eval_batchs = args.eval_batchs
        if self.dataset_format == 'petastorm':
            self.train_dataloader = make_petastorm_dataloader(args, "train")
        else:
            self.train_dataloader = make_data_loader(args, "train", self.is_global, self.dataset_format)
            self.val_dataloader = make_data_loader(args, "val", self.is_global, self.dataset_format)
        self.dlrm_module = make_dlrm_module(args)
        if self.is_global:
            self.dlrm_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
            self.dlrm_module.embedding.set_model_parallel(flow.env.all_device_placement("cuda"))
        else:
            self.dlrm_module.to("cuda")
        self.init_model()
        # self.opt = flow.optim.Adam(
        self.opt = flow.optim.SGD(
            self.dlrm_module.parameters(), lr=args.learning_rate
        )
        self.lr_scheduler = make_lr_scheduler(args, self.opt)
        if args.loss_scale_policy == "static":
            self.grad_scaler = flow.amp.StaticGradScaler(1024)
        else:
            self.grad_scaler = flow.amp.GradScaler(
                init_scale=1073741824,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )

        self.loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")
        if self.execution_mode == "graph":
            if self.dataset_format == 'petastorm':
                self.eval_graph = DLRMValGraph(self.dlrm_module, args.use_fp16)
                self.train_graph = DLRMTrainGraph(
                    self.dlrm_module, self.loss, self.opt, 
                    self.lr_scheduler, self.grad_scaler, args.use_fp16
                )
            else:
                self.eval_graph = DLRMValGraphWithDataloader(
                    self.dlrm_module, self.val_dataloader, args.use_fp16
                )
                self.train_graph = DLRMTrainGraphWithDataloader(
                    self.dlrm_module, self.train_dataloader, self.loss, self.opt, 
                    self.lr_scheduler, self.grad_scaler, args.use_fp16
                )

    def init_model(self):
        args = self.args
        if args.model_load_dir != "":
            self.load_state_dict()
        if self.ddp:
            self.dlrm_module = DDP(self.dlrm_module)
        if self.save_init and args.model_save_dir != "":
            self.save("initial_checkpoint")


    def load_state_dict(self):
        print(f"Loading model from {self.args.model_load_dir}")
        if self.is_global:
            state_dict = flow.load(self.args.model_load_dir, global_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.args.model_load_dir)
        else:
            return
        self.dlrm_module.load_state_dict(state_dict, strict=False)

    def save(self, subdir):
        if self.save_path is None or self.save_path == '':
            return
        save_path = os.path.join(self.save_path, subdir)
        if self.rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = self.dlrm_module.state_dict()
        if self.is_global:
            flow.save(state_dict, save_path, global_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return

    def __call__(self):
        self.train()

    def train(self):
        self.dlrm_module.train()
        last_iter, last_time = 0, time.time()
        for _ in range(self.max_iter):
            self.cur_iter += 1
            loss = self.train_one_step()

            if self.cur_iter % self.loss_print_every_n_iter == 0 and self.rank == 0:
                loss = tol(loss).mean().numpy()
                latency_ms = 1000 * (time.time() - last_time) / (self.cur_iter - last_iter)
                last_iter, last_time = 0, time.time()
                strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f'Iter {self.cur_iter}, Loss {loss:0.4f}, Latency_ms {latency_ms:0.3f}, {strtime}')

            if self.eval_interval > 0 and self.cur_iter % self.eval_interval == 0:
                self.eval(self.save_model_after_each_eval)
                last_time = time.time()

        if self.eval_after_training:
            self.eval(True)
            if self.args.eval_save_dir != '' and self.rank == 0:
                calculate_auc_from_dir(self.args.eval_save_dir)

    def eval(self, save_model=False):
        if self.eval_batchs <= 0:
            return
        self.dlrm_module.eval()
        if self.dataset_format == 'petastorm':
            self.val_dataloader = make_petastorm_dataloader(self.args, "val")
        labels = []
        preds = []
        for _ in range(self.eval_batchs):
            logits, label = self.inference()
            pred = logits.sigmoid()
            label_ = label.numpy().astype(np.float32)
            labels.append(label_)
            preds.append(pred.numpy())
        if self.args.eval_save_dir != '':
            if self.rank == 0:
                pf = os.path.join(self.args.eval_save_dir, f'iter_{self.cur_iter}.pkl')
                with open(pf, 'wb') as f:
                    obj = {'labels': labels, 'preds': preds, 'iter': self.cur_iter}
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            auc = roc_auc_score(label_, pred.numpy())
        else:
            labels = np.concatenate(labels, axis=0)
            preds = np.concatenate(preds, axis=0)
            auc = roc_auc_score(labels, preds)
        
        if self.rank == 0:
            strtime = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f'Iter {self.cur_iter}, AUC {auc:0.5f}, {strtime}')

        if save_model:
            sub_save_dir = f"iter_{self.cur_iter}_val_auc_{auc}"
            self.save(sub_save_dir)
        self.dlrm_module.train()

    def load_data(self, dataloader):
        labels, dense_fields, sparse_fields = dataloader()
        labels = labels.to("cuda")
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")
        return labels, dense_fields, sparse_fields

    def inference(self):
        if self.execution_mode == "graph":
            if self.dataset_format == "petastorm":
                labels, dense_fields, sparse_fields = self.load_data(self.val_dataloader)         
                return self.eval_graph(labels, dense_fields, sparse_fields)
            else:
                return self.eval_graph()
        else:
            labels, dense_fields, sparse_fields = self.load_data(self.val_dataloader)
            with flow.no_grad():
                predicts = self.dlrm_module(dense_fields, sparse_fields)
            return predicts, labels

    def train_one_step(self):
        self.dlrm_module.train()
        if self.execution_mode == "graph":
            if self.dataset_format == "petastorm":
                labels, dense_fields, sparse_fields = self.load_data(self.train_dataloader)
                return self.train_graph(labels, dense_fields, sparse_fields)
            else:
                return self.train_graph()
        else:
            labels, dense_fields, sparse_fields = self.load_data(self.train_dataloader)
            logits = self.dlrm_module(dense_fields, sparse_fields)
            loss = self.loss(logits, labels)
            loss = flow.mean(loss)
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            return loss


def tol(tensor, pure_local=True):
    """ to local """
    if tensor.is_global:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_global(sbp=flow.sbp.broadcast).to_local()

    return tensor


if __name__ == "__main__":
    flow.boxing.nccl.enable_all_to_all(True)
    trainer = Trainer()
    trainer()
