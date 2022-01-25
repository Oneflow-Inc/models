import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
from config import get_args
from models.data import make_data_loader,make_slot_loader
from models.dlrm import make_dlrm_module
from lr_scheduler import make_lr_scheduler
from oneflow.nn.parallel import DistributedDataParallel as DDP
from graph import DLRMValGraph, DLRMTrainGraph
import warnings
import utils.logger as log


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
        self.is_consistent = (
            flow.env.get_world_size() > 1 and not args.ddp
        ) or args.execution_mode == "graph"
        self.rank = flow.env.get_rank()
        self.world_size = flow.env.get_world_size()
        self.cur_iter = 0
        self.eval_interval = args.eval_interval
        self.eval_batchs = args.eval_batchs
        self.init_logger()
        self.train_dataloader = make_data_loader(args, "train", self.is_consistent, self.dataset_format)
        self.val_dataloader = make_data_loader(args, "val", self.is_consistent, self.dataset_format)
        self.slotloader = make_slot_loader(args.batch_size, args.eval_batch_size)
        self.dlrm_module = make_dlrm_module(args)
        
        if self.is_consistent:
            self.dlrm_module.to_consistent(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
        else:
            self.dlrm_module.to("cuda")
        
        self.init_model()

        # self.opt = flow.optim.Adam(
        self.opt = flow.optim.SGD(
            self.dlrm_module.parameters(), lr=args.learning_rate
        )
        self.lr_scheduler = make_lr_scheduler(args, self.opt)

        self.grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
        self.loss = flow.nn.BCELoss(reduction="none").to("cuda")
        if self.execution_mode == "graph":
            self.eval_graph = DLRMValGraph(
                self.dlrm_module, self.val_dataloader, self.slotloader
            )
            self.train_graph = DLRMTrainGraph(
                self.dlrm_module, self.train_dataloader, self.slotloader, self.loss, self.opt, self.lr_scheduler, self.grad_scaler
            )

    def init_model(self):
        args = self.args
        if args.model_load_dir != "":
            self.load_state_dict()
        if self.ddp:
            self.dlrm_module = DDP(self.dlrm_module)
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
        self.val_logger.register_metric("latency", log.LatencyMeter(), "latency(ms): {:.16f}", True)


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
        self.val_logger.meter("latency")
        if auc is not None:
            self.val_logger.meter("auc", auc)
        self.val_logger.print_metrics()


    def load_state_dict(self):
        """
        state_dict = {
            'bottom_mlp.linear_layers.fc0.features.0.bias': 'bot_l.0.bias.npy',
            'bottom_mlp.linear_layers.fc0.features.0.weight': 'bot_l.0.weight.npy',
            'bottom_mlp.linear_layers.fc1.features.0.bias': 'bot_l.2.bias.npy',
            'bottom_mlp.linear_layers.fc1.features.0.weight': 'bot_l.2.weight.npy',
            'bottom_mlp.linear_layers.fc2.features.0.bias': 'bot_l.4.bias.npy',
            'bottom_mlp.linear_layers.fc2.features.0.weight': 'bot_l.4.weight.npy',
            'embedding.weight': 'embedding_weight.npy',
            'top_mlp.linear_layers.fc0.features.0.bias': 'top_l.0.bias.npy',
            'top_mlp.linear_layers.fc0.features.0.weight': 'top_l.0.weight.npy',
            'top_mlp.linear_layers.fc1.features.0.bias': 'top_l.2.bias.npy',
            'top_mlp.linear_layers.fc1.features.0.weight': 'top_l.2.weight.npy',
            'top_mlp.linear_layers.fc2.features.0.bias': 'top_l.4.bias.npy',
            'top_mlp.linear_layers.fc2.features.0.weight': 'top_l.4.weight.npy',
            'top_mlp.linear_layers.fc3.features.0.bias': 'top_l.6.bias.npy',
            'top_mlp.linear_layers.fc3.features.0.weight': 'top_l.6.weight.npy',
            'scores.bias': 'top_l.8.bias.npy',
            'scores.weight': 'top_l.8.weight.npy',
        }
        for name, param in self.dlrm_module.named_parameters():
            path = os.path.join(self.args.model_load_dir, state_dict[name])
            W = np.load(path)
            print(name, param.shape, W.shape)
            #if eager
            #param.data = flow.tensor(W, requires_grad=True).to('cuda')
            #if graph:
            param.data = flow.tensor(W, requires_grad=True).to('cuda').to_consistent(placement = flow.placement("cuda", {0: 0}), sbp=flow.sbp.broadcast)
        # exit()
        return
        """
        if self.is_consistent:
            state_dict = flow.load(self.args.model_load_dir, consistent_src_rank=0)
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
        self.dlrm_module.train()
        for _ in range(self.max_iter):
            self.cur_iter += 1
            loss = self.train_one_step()

            loss = tol(loss)

            self.meter_train_iter(loss)

            if self.eval_interval > 0 and self.cur_iter % self.eval_interval == 0:
                self.eval(self.save_model_after_each_eval)
        if self.eval_after_training:
            self.eval(True)

    def eval(self, save_model=False):
        if self.eval_batchs <= 0:
            return
        self.dlrm_module.eval()
        labels = np.array([[0]])
        preds = np.array([[0]])
        for iter in range(self.eval_batchs):
            if self.execution_mode == "graph":
                pred, label = self.eval_graph()
            else:
                pred, label = self.inference()
            label_ = label.numpy().astype(np.float32)
            pred_ = pred.numpy()
            labels = np.concatenate((labels, label_), axis=0)
            preds = np.concatenate((preds, pred_), axis=0)
        auc = roc_auc_score(labels[1:], preds[1:])
        self.meter_eval(auc)
        if save_model:
            sub_save_dir = f"iter_{self.cur_iter}_val_auc_{auc}"
            self.save(sub_save_dir)
        self.dlrm_module.train()

    def inference(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.val_dataloader()
        sparse_slots = self.slotloader(is_train=False)
        labels = labels.to("cuda")
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")
        sparse_slots = sparse_slots.to("cuda")
        with flow.no_grad():
            predicts = self.dlrm_module(
                dense_fields, sparse_fields, sparse_slots
            )
        return predicts, labels

    def forward(self):
        (
            labels,
            dense_fields,
            sparse_fields,
        ) = self.train_dataloader()
        sparse_slots = self.slotloader(is_train=True)
        labels = labels.to("cuda")
        dense_fields = dense_fields.to("cuda")
        sparse_fields = sparse_fields.to("cuda")
        sparse_slots = sparse_slots.to("cuda")
        predicts = self.dlrm_module(dense_fields, sparse_fields, sparse_slots)
        loss = self.loss(predicts, labels)
        reduce_loss = flow.mean(loss)
        return reduce_loss

    def train_eager(self):
        loss = self.forward()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def train_one_step(self):
        self.dlrm_module.train()
        if self.execution_mode == "graph":
            train_loss = self.train_graph()
        else:
            train_loss = self.train_eager()
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
    flow.boxing.nccl.enable_all_to_all(True)
    trainer = Trainer()
    trainer()
