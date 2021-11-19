import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
from config import get_args
from models.data import make_data_loader
from models.wide_and_deep import make_wide_and_deep_module
from oneflow.nn.parallel import DistributedDataParallel as DDP
from graph import WideAndDeepValGraph, WideAndDeepTrainGraph
import warnings
import utils.logger as log


class Trainer(object):
    def __init__(self):
        args = get_args()
        self.args = args
        self.save_path = args.model_save_dir
        self.save_init = args.save_initial_model
        self.save_model_after_each_eval = args.save_model_after_each_eval
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
        self.is_consistent = (
            flow.env.get_world_size() > 1 and not args.ddp
        ) or args.execution_mode == "graph"
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.cur_iter = 0
        self.eval_interval = args.eval_interval
        self.eval_batchs = args.eval_batchs
        self.init_logger()
        self.train_dataloader = make_data_loader(args, "train", self.is_consistent)
        self.val_dataloader = make_data_loader(args, "val", self.is_consistent)
        self.wdl_module = make_wide_and_deep_module(args)
        self.init_model()
        self.opt = flow.optim.AdamW(
            self.wdl_module.parameters(), lr=args.learning_rate
        )
        self.loss = flow.nn.BCELoss(reduction="none").to("cuda")
        if self.execution_mode == "graph":
            self.eval_graph = WideAndDeepValGraph(
                self.wdl_module, self.val_dataloader
            )
            self.train_graph = WideAndDeepTrainGraph(
                self.wdl_module, self.train_dataloader, self.loss, self.opt
            )

    def init_model(self):
        args = self.args
        if self.is_consistent == True:
            placement = placement = flow.env.all_device_placement("cuda")
            self.wdl_module = self.wdl_module.to_consistent(
                placement=placement, sbp=flow.sbp.broadcast
            )
        else:
            self.wdl_module = self.wdl_module.to("cuda")
        if args.model_load_dir != "":
            self.load_state_dict()
        if self.ddp:
            self.wdl_module = DDP(self.wdl_module)
        if self.save_init and args.model_save_dir != "":
            self.save(os.path.join(args.model_save_dir, "initial_checkpoint"))
    
    def init_logger(self):
        print_ranks = [0]
        self.train_logger = log.make_logger(self.rank, print_ranks)
        self.train_logger.register_metric("iter", log.IterationMeter(), "iter: {}/{}")
        self.train_logger.register_metric("loss", log.AverageMeter(), "loss: {:.16f}", True)
        self.train_logger.register_metric("latency", log.LatencyMeter(), "latency(ms): {:.16f}", True)

        self.val_logger = log.make_logger(self.rank, print_ranks)
        self.val_logger.register_metric("iter", log.IterationMeter(), "iter: {}/{}")
        self.val_logger.register_metric("auc", log.IterationMeter(), "eval_auc: {}")
    
    def meter(
        self,
        loss=None,
        do_print=False,
    ):
        self.train_logger.meter("iter", (self.cur_iter, self.max_iter))
        if loss is not None:
            self.train_logger.meter("loss", loss)
        self.train_logger.meter("latency")
        if do_print:
            self.train_logger.print_metrics()

    def meter_train_iter(self, loss):
        do_print = (
            self.cur_iter % self.loss_print_every_n_iter == 0
        )
        self.meter(
            loss=loss,
            do_print=do_print,
        )
    
    def meter_eval(self, auc):
        self.val_logger.meter("iter", (self.cur_iter, self.max_iter))
        if auc is not None:
            self.val_logger.meter("auc", auc)
        self.val_logger.print_metrics()
  

    def load_state_dict(self):
        print(f"Loading model from {self.args.model_load_dir}")
        if self.is_consistent:
            state_dict = flow.load(self.args.model_load_dir, consistent_src_rank=0)
        elif self.rank == 0:
            state_dict = flow.load(self.args.model_load_dir)
        else:
            return
        self.wdl_module.load_state_dict(state_dict)

    def save(self, subdir):
        if self.save_path is None:
            return
        save_path = os.path.join(self.save_path, subdir)
        if self.rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = self.wdl_module.state_dict()
        if self.is_consistent:
            flow.save(state_dict, save_path, consistent_dst_rank=0)
        elif self.rank == 0:
            flow.save(state_dict, save_path)
        else:
            return

    def __call__(self):
        self.train()

    def train(self):
        losses = []
        self.wdl_module.train()
        for i in range(self.max_iter):
            self.cur_iter += 1
            loss = self.train_one_step()
            
            if self.ddp:
                # In ddp mode, the loss needs to be averaged
                loss = flow.comm.all_reduce(loss)
                loss = loss / self.world_size
            
            loss = tol(loss, False)

            self.meter_train_iter(loss)
            
            if self.eval_interval > 0 and (i + 1) % self.eval_interval == 0:
                self.eval(self.save_model_after_each_eval)
        self.eval(True)     
    
    def eval(self, save_model=False):
        self.wdl_module.eval()
        labels = np.array([[0]])
        preds = np.array([[0]])
        for _ in range(self.eval_batchs):
            if self.execution_mode == "graph":
                pred, label = self.eval_graph()
            else:
                pred, label = self.inference()
            label_ = label.numpy().astype(np.float32)
            labels = np.concatenate((labels, label_), axis=0)
            preds = np.concatenate((preds, pred.numpy()), axis=0)
        auc = roc_auc_score(labels[1:], preds[1:])
        self.meter_eval(auc)
        if save_model:
            sub_save_dir = f"iter_{self.cur_iter}_val_auc_{auc}"
            self.save(sub_save_dir)
        self.wdl_module.train()

    def inference(self):
        (
            labels,
            dense_fields,
            wide_sparse_fields,
            deep_sparse_fields,
        ) = self.val_dataloader()
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")
        with flow.no_grad():
            logits = self.wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields
            )
            predicts = flow.sigmoid(logits)
        return predicts, labels

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
        return predicts, labels, reduce_loss

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
