import argparse
import os
import sys
import glob
import time
import psutil
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from sklearn.metrics import roc_auc_score

from utils import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader

from tqdm import tqdm


num_dense_fields = 0
num_sparse_fields = 10

def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=10)
    parser.add_argument("--dnn_use_bn", type=bool, default=True)
    # parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--cross_parameterization", type=str, default="vector")
    parser.add_argument("--dnn_activation", type=str, default="relu")
    parser.add_argument("--dnn_hidden_units", type=int_list, default="400,400,400")
    parser.add_argument("--embedding_dim", type=int, default=10)
    # parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loss", type=str, default="binary_crossentropy")
    parser.add_argument("--metrics", type=str, default="binary_crossentropy,auc")
    parser.add_argument("--dnn_dropout", type=float, default=0.2)
    parser.add_argument("--l2_reg_embedding", type=int, default=0.005)
    parser.add_argument("--l2_reg_cross", type=float, default=0.00001)
    parser.add_argument("--l2_reg_dnn", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    # parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument(
        "--disable_interaction_padding",
        action="store_true",
        help="disable interaction padding or not",
    )
    parser.add_argument(
        "--interaction_itself", action="store_true", help="interaction itself or not"
    )
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not.",
    )
    parser.add_argument(
        "--save_model_after_each_eval", action="store_true", help="save model after each eval.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--eval_batches", type=int, default=113, help="number of eval batches")
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--warmup_batches", type=int, default=500)
    parser.add_argument("--decay_batches", type=int, default=1000)
    parser.add_argument("--decay_start", type=int, default=2500)
    parser.add_argument("--train_batches", type=int, default=5000)
    parser.add_argument("--loss_print_interval", type=int, default=50)
    parser.add_argument(
        "--table_size_array",
        type=int_list,
        help="Embedding table size array for sparse fields",
        required=True,
    )
    parser.add_argument(
        "--persistent_path", type=str, required=True, help="path for persistent kv store",
    )
    parser.add_argument("--store_type", type=str, default="cached_host_mem")
    parser.add_argument("--cache_memory_budget_mb", type=int, default=2048)
    parser.add_argument("--amp", action="store_true", help="Run model with amp")
    parser.add_argument("--loss_scale_policy", type=str, default="static", help="static or dynamic")

    args = parser.parse_args()

    if print_args and flow.env.get_rank() == 0:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)



class FrappeDataReader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs,
        shuffle_row_groups=True,
        shard_seed=1234,
        shard_count=1,
        cur_shard=0,
    ):
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_row_groups = shuffle_row_groups
        self.shard_seed = shard_seed
        self.shard_count = shard_count
        self.cur_shard = cur_shard

        fields = ["label"]
        fields += []
        self.I_end = len(fields)
        fields += ["user","item","daytime","weekday","isweekend","homework","cost","weather","country","city"]
        self.C_end = len(fields)
        self.fields = fields

    def __enter__(self):
        self.reader = make_batch_reader(
            self.parquet_file_url_list,
            workers_count=2,
            shuffle_row_groups=self.shuffle_row_groups,
            num_epochs=self.num_epochs,
            shard_seed=self.shard_seed,
            shard_count=self.shard_count,
            cur_shard=self.cur_shard,
        )
        self.loader = self.get_batches(self.reader)
        return self.loader

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.stop()
        self.reader.join()

    def get_batches(self, reader, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        tail = None
        for rg in reader:
            rgdict = rg._asdict()
            rglist = [rgdict[field] for field in self.fields]
            pos = 0
            if tail is not None:
                pos = batch_size - len(tail[0])
                tail = list(
                    [
                        np.concatenate((tail[i], rglist[i][0 : (batch_size - len(tail[i]))]))
                        for i in range(self.C_end)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    dense = tail[1 : self.I_end]
                    sparse = tail[self.I_end : self.C_end]
                    tail = None
                    # yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
                    yield label, np.stack(sparse, axis=-1)

                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                # dense = [rglist[j][pos : pos + batch_size] for j in range(1, self.I_end)]
                sparse = [rglist[j][pos : pos + batch_size] for j in range(self.I_end, self.C_end)]
                pos += batch_size
                # yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
                yield label, np.stack(sparse, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.C_end)]



def make_frappe_dataloader(data_path, batch_size, shuffle=True):
    """Make a Frappe_x1 Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return FrappeDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=1234,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )

def batch_to_global(np_label, np_sparse):
# def batch_to_global(np_label, np_dense, np_sparse):
    def _np_to_global(np, dtype=flow.float):
        t = flow.tensor(np, dtype=dtype)
        return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))

    labels = _np_to_global(np_label.reshape(-1, 1))
    # dense_fields = _np_to_global(np_dense)
    sparse_fields = _np_to_global(np_sparse, dtype=flow.int64)
    # return labels, dense_fields, sparse_fields
    return labels, sparse_fields

class OneEmbedding(nn.Module):
    def __init__(
        self,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        store_type,
        cache_memory_budget_mb,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table(
                flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
            )
            for scale in scales
        ]
        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, capacity=vocab_size
            )
        elif store_type == "cached_host_mem":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_host_mem_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
            )
        elif store_type == "cached_ssd":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
            )
        else:
            raise NotImplementedError("not support", store_type)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableEmbedding(
            "sparse_embedding",
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)

class CrossNet(nn.Module):
    def __init__(
        self,
        in_features,
        layer_num=2,
        parameterization="vector",
    ):
        super(CrossNet, self).__init__()

        self.layer_num = layer_num
        self.parameterization = parameterization

        if self.parameterization == "vector":
            # weight in DCN.  (in_features, 1)
            self.kernels = nn.Parameter(flow.Tensor(self.layer_num, in_features, 1))
        elif self.parameterization == "matrix":
            # weight matrix in DCN-M.  (in_features, in_features)
            self.kernels = nn.Parameter(
                flow.Tensor(self.layer_num, in_features, in_features)
            )
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")

        self.bias = nn.Parameter(flow.Tensor(self.layer_num, in_features, 1))

        nn.init.xavier_normal_(self.kernels)
        nn.init.zeros_(self.bias)


    def forward(self, inputs):
        x_0 = inputs.unsqueeze(2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == "vector":
                xl_w = flow.einsum("abc,bd->acd", x_l, self.kernels[i])
                dot_ = flow.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == "matrix":
                xl_w = flow.matmul(self.kernels[i], x_l)  # W * xi  (bs, in_features, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = flow.squeeze(x_l, dim=2)
        return x_l



class DNN(nn.Module):
    def __init__(
        self,
        inputs_dim,
        hidden_units,
        activation="relu",
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        init_std=0.0001,
        dice_dim=3,
    ):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [
                nn.Linear(hidden_units[i], hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ]
        )

        if self.use_bn:
            self.bn = nn.ModuleList(
                [
                    nn.BatchNorm1d(hidden_units[i + 1])
                    for i in range(len(hidden_units) - 1)
                ]
            )

        self.activation_layers = nn.ModuleList(
            [
                activation_layer(activation, hidden_units[i + 1], dice_dim)
                for i in range(len(hidden_units) - 1)
            ]
        )

        for name, tensor in self.linears.named_parameters():
            if "weight" in name:
                nn.init.normal_(tensor, mean=0, std=init_std)


    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc

        return deep_input



class DCNModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size,

        persistent_path,
        table_size_array,
        one_embedding_store_type,
        cache_memory_budget_mb,

        cross_num=2,
        cross_parameterization="vector",
        dnn_hidden_units=(128, 128),
        l2_reg_linear=0.00001,
        l2_reg_cross=0.00001,
        l2_reg_dnn=0,
        init_std=0.0001,
        dnn_dropout=0,
        dnn_activation="relu",
        dnn_use_bn=False,
    ):

        super(DCNModule, self).__init__()

        self.reg_loss = flow.zeros((1,))
        self.aux_loss = flow.zeros((1,))

        self.dnn_hidden_units = dnn_hidden_units
        self.cross_num = cross_num

        ### 
        self.embedding_layer = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            table_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb,
        )
        ###


        # self.compute_input_dim = compute_input_dim
        # 10*(0+10) = 100
        input_dim = embedding_vec_size * (num_dense_fields + num_sparse_fields) 
        # print(input_dim) 

        self.dnn = DNN(
            input_dim,
            dnn_hidden_units,
            activation=dnn_activation,
            use_bn=dnn_use_bn,
            l2_reg=l2_reg_dnn,
            dropout_rate=dnn_dropout,
            init_std=init_std,
        )


        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:
            last_linear_in_feature = input_dim + dnn_hidden_units[-1]
        elif len(self.dnn_hidden_units) > 0:
            last_linear_in_feature = dnn_hidden_units[-1]
        elif self.cross_num > 0:
            last_linear_in_feature = input_dim

        # self.last_linear = nn.Linear(last_linear_in_feature, 1, bias=False).to(device)
        self.last_linear = nn.Linear(last_linear_in_feature, 1, bias=False)

        self.crossnet = CrossNet(
            in_features=input_dim,
            layer_num=cross_num,
            parameterization=cross_parameterization,
        )


        self.regularization_weight = []
        self.add_regularization_weight(
            filter(
                lambda x: "weight" in x[0] and "bn" not in x[0],
                self.dnn.named_parameters(),
            ),
            l2=l2_reg_dnn,
        )
        self.add_regularization_weight(self.last_linear.weight, l2=l2_reg_linear)
        self.add_regularization_weight(self.crossnet.kernels, l2=l2_reg_cross)

        # self.to(device)


    def forward(self, X) -> flow.Tensor:

        embedded_x = self.embedding_layer(X)
        if len(self.dnn_hidden_units) > 0 and self.cross_num > 0:  # Deep & Cross
            # print("aaaaaaaaaaaaaaaaaa")
            deep_out = self.dnn(embedded_x.flatten(start_dim=1))
            # print(deep_out.shape)
            cross_out = self.crossnet(embedded_x.flatten(start_dim=1))
            # print(cross_out.shape)
            stack_out = flow.cat((cross_out, deep_out), dim=-1)
            logit = self.last_linear(stack_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(embedded_x.flatten(start_dim=1))
            logit = self.last_linear(deep_out)
        elif self.cross_num > 0:  # Only Cross
            cross_out = self.crossnet(embedded_x.flatten(start_dim=1))
            logit = self.last_linear(cross_out)
        else:  # Error
            raise Exception("Model must be Deep & Cross, Only Deep or Only Cross.")
        y_pred = activation_layer("sigmoid")(logit)
        return y_pred
    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, flow.nn.Parameter):
        # if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        # total_reg_loss = flow.zeros((1,), device=self.device)
        total_reg_loss = flow.zeros((1,))   
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += flow.sum(l1 * flow.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += flow.sum(l2 * flow.square(parameter))
                    except AttributeError:
                        total_reg_loss += flow.sum(l2 * parameter * parameter)

        return total_reg_loss


    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

def make_dcn_module(args):
    model = DCNModule(
                embedding_vec_size=args.embedding_vec_size,

                persistent_path=args.persistent_path,
                table_size_array=args.table_size_array,
                one_embedding_store_type=args.store_type,
                cache_memory_budget_mb=args.cache_memory_budget_mb,

                cross_num=args.cross_num, 
                cross_parameterization=args.cross_parameterization,               
                dnn_hidden_units=args.dnn_hidden_units,  
                l2_reg_cross=args.l2_reg_cross,
                l2_reg_dnn=args.l2_reg_dnn, 
                dnn_dropout=args.dnn_dropout, 
                dnn_activation=args.dnn_activation, 
                dnn_use_bn=args.dnn_use_bn,                 
            )
    return model

class DCNValGraph(flow.nn.Graph):
    def __init__(self, deepfm_module, amp=False):
        super(DCNValGraph, self).__init__()
        self.module = deepfm_module
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        predicts = self.module(features.to("cuda"))
        return predicts.to("cpu")


class DCNTrainGraph(flow.nn.Graph):
    def __init__(
        self, deepfm_module, loss, optimizer, lr_scheduler=None, grad_scaler=None, amp=False,
    ):
        super(DCNTrainGraph, self).__init__()
        self.module = deepfm_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, features):
        
        logits = self.module(features.to("cuda")).squeeze()
        loss = self.loss(logits, labels.squeeze().to("cuda"))

        # reg_loss = self.module.get_regularization_loss().to("cuda")

        # total_loss = loss + reg_loss + self.module.aux_loss.to("cuda")
        total_loss = loss

        reduce_loss = flow.mean(total_loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")

def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=args.warmup_batches,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, steps=args.decay_batches, end_learning_rate=0, power=2.0, cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[args.decay_start],
        interval_rescaling=True,
    )
    return sequential_lr


def eval(args, eval_graph, cur_step=0):
    if args.eval_batches <= 0:
        return
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    with make_frappe_dataloader(f"{args.data_dir}/test", args.eval_batch_size, shuffle=False) as loader:

        num_eval_batches = 0
        for np_batch in loader:
            num_eval_batches += 1
            if num_eval_batches > args.eval_batches:
                break
            label, features = batch_to_global(*np_batch)
            pred = eval_graph(features) # Caution: sigmoid in module or only in eval?
            labels.append(label.numpy())
            preds.append(pred.numpy())

    auc = 0  # will be updated by rank 0 only
    rank = flow.env.get_rank()
    if rank == 0:
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        eval_time = time.time() - eval_start_time
        auc_start_time = time.time()
        auc = roc_auc_score(labels, preds)
        auc_time = time.time() - auc_start_time

        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Step {cur_step}, AUC {auc:0.5f}, Eval_time {eval_time:0.2f} s, "
            + f"AUC_time {auc_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    flow.comm.barrier()
    return auc


def train(args):
    rank = flow.env.get_rank()
    dcn_module = make_dcn_module(args)
    dcn_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)


    optim = flow.optim.SGD(dcn_module.parameters(), lr=args.learning_rate)
    lr_scheduler = make_lr_scheduler(args, optim)
    loss_func = nn.BCELoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )
    
    eval_graph = DCNValGraph(dcn_module, args.amp)
    train_graph = DCNTrainGraph(dcn_module, loss_func, optim, lr_scheduler, grad_scaler, args.amp)
    
    dcn_module.train()

    step, last_step, last_time = -1, 0, time.time()
    with make_frappe_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        for step in range(1, args.train_batches + 1):
            # labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            labels, sparse_fields = batch_to_global(*next(loader))
            loss = train_graph(labels, sparse_fields)

            if step % args.loss_print_interval == 0:

                loss = loss.numpy()
                if rank == 0:
                    latency_ms = 1000 * (time.time() - last_time) / (step - last_step)
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {latency_ms:0.3f} ms, {strtime}"
                    )

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                auc = eval(args, eval_graph, step)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{auc:0.5f}")
                dcn_module.train()
                last_time = time.time()


    if args.eval_interval > 0 and step % args.eval_interval != 0:
        auc = eval(args, eval_graph, step)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)