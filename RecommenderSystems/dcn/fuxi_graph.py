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
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader

from tqdm import tqdm




num_dense_fields = 0
num_sparse_fields = 10

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        else:
            return getattr(nn, activation)()
    else:
        return activation

def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()


    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=10)
    parser.add_argument("--batch_norm", type=bool, default=True)
    # parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--crossing_layers", type=int, default=3)
    parser.add_argument("--cross_parameterization", type=str, default="vector")
    parser.add_argument("--net_dropout", type=float, default=0.2)
    parser.add_argument("--dnn_activations", type=str, default="relu")
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

    parser.add_argument("--reduce_lr_on_plateau", type=bool, default=True)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1.0e-6)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)

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
    parser.add_argument("--size_factor", type=int, default=1)

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
        size_factor,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        for i in range(len(table_size_array)):
            tables = [flow.one_embedding.make_table(flow.one_embedding.make_normal_initializer(0, 1e-4))]
            # tables = [flow.one_embedding.make_table(flow.one_embedding.make_normal_initializer(0, 1))]
        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, capacity=vocab_size, size_factor=size_factor
            )
        elif store_type == "cached_host_mem":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_host_mem_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
            )
        elif store_type == "cached_ssd":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
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

class CrossInteractionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(flow.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(CrossInteractionLayer(input_dim)
                                       for _ in range(self.num_layers))

    def forward(self, X_0):
        X_i = X_0 # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class DNN(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim=None, 
                 hidden_units=[], 
                 hidden_activations="ReLU",
                 output_activation=None, 
                 dropout_rates=[], 
                 batch_norm=False, 
                 use_bias=True):
        super(DNN, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm:
                print("batch_normbatch_normbatch_normbatch_normbatch_normbatch_normbatch_normbatch_norm")
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.dnn = nn.Sequential(*dense_layers) # * used to unpack list
    
    def forward(self, inputs):
        return self.dnn(inputs)


class DCNModule(nn.Module):
    def __init__(self, 
        embedding_vec_size,
        persistent_path,
        table_size_array,
        one_embedding_store_type,
        cache_memory_budget_mb,
        size_factor,

        dnn_hidden_units=[128, 128],

        crossing_layers = 3,
        net_dropout = 0.2,
        dnn_activations="relu",
        batch_norm=False,
        ):
        super(DCNModule, self).__init__()


        # self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.embedding_layer = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            table_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb,
            size_factor=size_factor
        )

        input_dim = embedding_vec_size * (num_dense_fields + num_sparse_fields) 

        self.dnn = DNN(input_dim=input_dim,
                             output_dim=None, # output hidden layer
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=None, 
                             dropout_rates=net_dropout, 
                             batch_norm=batch_norm, 
                             use_bias=True) \
                   if dnn_hidden_units else None # in case of only crossing net used

        self.crossnet = CrossNet(input_dim, crossing_layers)


        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0: # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1) # [cross_part, dnn_part] -> logit

        self.output_activation = nn.Sigmoid()
        self.reset_parameters()
        # self.model_to_device()

    def forward(self, X):

        feature_emb = self.embedding_layer(X)
        flat_feature_emb = feature_emb.flatten(start_dim=1)
        cross_out = self.crossnet(flat_feature_emb)
        if self.dnn is not None:
            dnn_out = self.dnn(flat_feature_emb)
            final_out = flow.cat([cross_out, dnn_out], dim=-1)
        else:
            final_out = cross_out
        y_pred = self.fc(final_out)
        y_pred = self.output_activation(y_pred)
        return y_pred

    def reset_parameters(self):
        def reset_param(m):
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
        self.apply(reset_param)




def make_dcn_module(args):
    model = DCNModule(
                embedding_vec_size=args.embedding_vec_size,

                persistent_path=args.persistent_path,
                table_size_array=args.table_size_array,
                one_embedding_store_type=args.store_type,
                cache_memory_budget_mb=args.cache_memory_budget_mb,
                dnn_hidden_units = args.dnn_hidden_units,
                crossing_layers=args.crossing_layers, 
                net_dropout = args.net_dropout,
                dnn_activations=args.dnn_activations, 
                batch_norm=args.batch_norm,   
                size_factor=args.size_factor         
            )
    # print(model)
    # print(model.state_dict())
    for key in model.state_dict():
        print(key)
        # print(model.state_dict()[key])
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
        total_loss = loss 
        # total_loss = loss + self.add_regularization()

        reduce_loss = flow.mean(total_loss)
        reduce_loss.backward()

        return reduce_loss.to("cpu")
#     def add_regularization(self):
#         reg_loss = 0

#         net_reg = get_regularizer(0.001)
#         print("==================================")
#         for name, param in self.module.named_parameters():
#             if "embedding" in name :
#                 continue
#             print(name)
#             print(type(param))
#             print(param)
#             for i in param:
#                 print(i)
#             if param.requires_grad:
#                 for net_p, net_lambda in net_reg:
#                     reg_loss += (net_lambda / net_p) * flow.norm(param, net_p) ** net_p
#         return reg_loss

# def get_regularizer(reg):
#     reg_pair = [] # of tuples (p_norm, weight)
#     if isinstance(reg, float):
#         reg_pair.append((2, reg))
#     elif isinstance(reg, str):
#         try:
#             if reg.startswith("l1(") or reg.startswith("l2("):
#                 reg_pair.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
#             elif reg.startswith("l1_l2"):
#                 l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
#                 reg_pair.append((1, float(l1_reg)))
#                 reg_pair.append((2, float(l2_reg)))
#             else:
#                 raise NotImplementedError
#         except:
#             raise NotImplementedError("regularizer={} is not supported.".format(reg))
#     return reg_pair


def make_lr_scheduler(args, optimizer):
    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)
    milestones = [batches_per_epoch * (i + 1) for i in range(math.floor(math.log(args.min_lr / args.learning_rate, args.lr_factor)))]
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        gamma=args.lr_factor,
        milestones=milestones,
    )
    return multistep_lr






def eval(args, eval_graph, cur_step=0, epoch=0):
    if args.eval_batches <= 0:
        return
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    # with make_frappe_dataloader(f"{args.data_dir}/test", args.eval_batch_size, shuffle=False) as loader:
    with make_frappe_dataloader(f"{args.data_dir}/val", args.eval_batch_size, shuffle=False) as loader:
        num_eval_batches = 0
        for np_batch in loader:
            num_eval_batches += 1
            if num_eval_batches > args.eval_batches:
                break
            label, features = batch_to_global(*np_batch)
            logits = eval_graph(features)
            pred = logits
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
            f"Rank[{rank}], Epoch {epoch}, Step {cur_step}, AUC {auc:0.5f}, Eval_time {eval_time:0.2f} s, "
            + f"AUC_time {auc_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    flow.comm.barrier()
    return auc




def train(args):
    rank = flow.env.get_rank()
    dcn_module = make_dcn_module(args)
    dcn_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    if args.model_load_dir:
        print(f"Loading model from {args.model_load_dir}")
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        dcn_module.load_state_dict(state_dict, strict=False)
    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = dcn_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)
    # optim = flow.optim.SGD(dcn_module.parameters(), lr=args.learning_rate)
    optim = flow.optim.Adam(dcn_module.parameters(), lr=args.learning_rate)

    def lr_decay(factor=0.1, min_lr=1e-6):
            for param_group in optim.param_groups:
                reduced_lr = max(param_group["lr"] * factor, min_lr)
                param_group["lr"] = reduced_lr
            print("lr_decay: {}".format(reduced_lr))
            return reduced_lr

    def early_stop(epoch, logs, best_metric, stopping_steps, 
                    patience=2, min_delta=1e-6, _reduce_lr_on_plateau=True):
        kv = {'auc': 1, 'logloss': -1}
        monitor_value = 0
        for k, v in kv.items():
            monitor_value += logs.get(k, 0) * v

        stop_training = False
        if monitor_value < best_metric + min_delta:
                stopping_steps += 1
                print("Monitor(max) STOP: {:.6f}!".format(monitor_value))
                if _reduce_lr_on_plateau:
                    lr_decay(factor=0.1, min_lr=1e-6)
        else:
            stopping_steps = 0
            best_metric = monitor_value
        # if stopping_steps >= patience:
        #     stop_training = False
        #     print(f"Early stopping at epoch={epoch}!")
        return stop_training, best_metric, stopping_steps

    # lr_scheduler = make_lr_scheduler(args, optim)
    lr_scheduler = None
    args.num_train_samples = 202027
    # lr_scheduler = make_lr_scheduler(args, optim)
    loss_func = nn.BCELoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )
    
    eval_graph = DCNValGraph(dcn_module, args.amp)
    train_graph = DCNTrainGraph(dcn_module, loss_func, optim, lr_scheduler, grad_scaler, args.amp)



    train_losses = []
    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)

    batches_per_epoch = 25  
    best_metric = -np.inf
    stopping_steps = 0

    dcn_module.train()
    step, last_step, last_time = -1, 0, time.time()
    epoch = 0
    with make_frappe_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        for step in range(1, args.train_batches + 1):
            # labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            labels, sparse_fields = batch_to_global(*next(loader))
            # if step==2:
            #     labels = labels[:10]
            #     sparse_fields = sparse_fields[:10]
            # print(labels.shape)
            # print(sparse_fields.shape)

            # if step ==2:
            #     break
            loss = train_graph(labels, sparse_fields)



            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                train_losses.append(loss)
                if rank == 0:
                    latency_ms = 1000 * (time.time() - last_time) / (step - last_step)
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {latency_ms:0.3f} ms, {strtime}"
                    )


            if step % batches_per_epoch == 0:
                print(optim.state_dict()['param_groups'][0]['_options']["lr"] )
                epoch += 1
                auc = eval(args, eval_graph, step, epoch)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{auc:0.5f}")

                stop_training, best_metric, stopping_steps = early_stop(
                    epoch, 
                    logs={'auc': auc}, 
                    best_metric=best_metric, 
                    stopping_steps=stopping_steps,


                    patience=args.patience, 
                    min_delta=args.min_delta,
                    _reduce_lr_on_plateau = args.reduce_lr_on_plateau
                )
                
                if stop_training:
                    break

                dcn_module.train()
                last_time = time.time()

    if step % batches_per_epoch != 0:
        auc = eval(args, eval_graph, step)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")

    plot_train_curve(args,train_losses)
    #         if args.eval_interval > 0 and step % args.eval_interval == 0:
    #             auc = eval(args, eval_graph, step)
    #             if args.save_model_after_each_eval:
    #                 save_model(f"step_{step}_val_auc_{auc:0.5f}")
    #             dcn_module.train()
    #             last_time = time.time()


    # if args.eval_interval > 0 and step % args.eval_interval != 0:
    #     auc = eval(args, eval_graph, step)
    #     if args.save_model_after_each_eval:
    #         save_model(f"step_{step}_val_auc_{auc:0.5f}")

def plot_train_curve(args, train_losses):
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.ylabel("Training Logloss")
    plt.xlabel(f"batch number (per {args.loss_print_interval} batches)")
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)
