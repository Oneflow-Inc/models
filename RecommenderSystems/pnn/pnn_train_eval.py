import argparse
import os
import sys
import glob
import time
import math
import numpy as np
import psutil
import warnings
import oneflow as flow
import oneflow.nn as nn
from sklearn.metrics import roc_auc_score, log_loss
from petastorm.reader import make_batch_reader
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_train_samples", type=int, required=True, help="the number of training samples")

    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument("--save_initial_model", action="store_true", help="save initial model parameters or not.")
    parser.add_argument("--save_model_after_each_eval", action="store_true", help="save model after each eval.")

    parser.add_argument("--disable_fusedmlp", default=False, help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=16)
    parser.add_argument("--dnn", type=int_list, default="1000,1000,1000,1000,1000")
    parser.add_argument("--net_dropout", type=float, default=0.2)
    parser.add_argument("--embedding_regularizer", type=float, default=None)
    parser.add_argument("--net_regularizer", type=float, default=None)
    parser.add_argument("--max_gradient_norm", type=float, default=10.0)

    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    
    parser.add_argument("--eval_batches", type=int, default=1612, help="number of eval batches")
    parser.add_argument("--eval_batch_size", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=10000)
    parser.add_argument("--train_batches", type=int, default=75000)
    parser.add_argument("--loss_print_interval", type=int, default=1000)

    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1.0e-6)
    
    parser.add_argument(
        "--table_size_array",
        type=int_list,
        help="Embedding table size array for sparse fields",
        required=True,
    )
    parser.add_argument("--persistent_path", type=str, required=True, help="path for persistent kv store")
    parser.add_argument("--store_type", type=str, default="cached_host_mem")
    parser.add_argument("--cache_memory_budget_mb", type=int, default=8192)

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


num_dense_fields = 13
num_sparse_fields = 26


class PNNDataReader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs=1,
        shuffle_row_groups=True,
        shard_seed=2020,
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

        fields = ["Label"]
        fields += [f"I{i+1}" for i in range(num_dense_fields)]
        fields += [f"C{i+1}" for i in range(num_sparse_fields)]
        self.fields = fields
        self.num_fields = len(fields)

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
                        for i in range(self.num_fields)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    features = tail[1:]
                    tail = None
                    features = np.stack(features, axis=-1)
                    yield label, features
                else:
                    pos = 0
                    continue

            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                # TODO: check list slicing failed problem
                tmp = np.array(rglist)
                # features = rglist[1:][pos: pos + batch_size]
                features = tmp[1:, pos: pos + batch_size]
                pos += batch_size
                features = np.stack(features, axis=-1)
                yield label, features

            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.num_fields)]


def make_criteo_dataloader(data_path, batch_size, shuffle=True):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return PNNDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=2020,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )


class OneEmbedding(nn.Module):
    def __init__(
        self,
        table_name,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        store_type,
        cache_memory_budget_mb,
        size_factor,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        scales = np.sqrt(1 / np.array(table_size_array))
        # TODO: embedding initialization uniform2normal
        tables = [
            flow.one_embedding.make_table(
                # flow.one_embedding.make_uniform_initializer(low=-scale, high=scale)
                flow.one_embedding.make_normal_initializer(mean=0.0, std=1e-4)
            )
            # for scale in scales
            for _ in range(len(table_size_array))
        ]
        # tables = [
        #     flow.one_embedding.make_table(
        #         flow.one_embedding.make_normal_initializer(mean=0.0, std=1e-4)
        #     )
        #     for _ in range(len(table_size_array))
        # ]
        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, 
                capacity=vocab_size,
                size_factor=size_factor,
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
            name=table_name,
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class DenseLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, relu=True, dropout=0.0) -> None:
        super(DenseLayer, self).__init__()
        denses = []
        denses.append(nn.Linear(in_features, out_features))
        if relu:
            denses.append(nn.ReLU(inplace=True))
        if dropout > 0:
            denses.append(nn.Dropout(p=dropout))
        self.features = nn.Sequential(*denses)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.features(x)


class DNN(nn.Module):
    def __init__(self, in_features: int, hidden_units, skip_final_activation=False, fused=True, dropout=0.0) -> None:
        super(DNN, self).__init__()
        if fused: # TODO: add dropout in Fused MLP
            self.linear_layers = nn.FusedMLP(
                in_features,
                hidden_units[:-1],
                hidden_units[-1],
                skip_final_activation=skip_final_activation
            )
        else:
            # TODO: support different dropout rates for each layer
            dropout_rates = [dropout] * (len(hidden_units) - 1) + [0.0]
            use_relu = [True] * (len(hidden_units) - 1) + [not skip_final_activation]
            units = [in_features] + hidden_units
            denses = [
                DenseLayer(units[i], units[i + 1], relu=use_relu[i], dropout=dropout_rates[i]) 
                for i in range(len(units) - 1)
            ]
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


# class LR(nn.Module):
#     def __init__(
#         self,
#         persistent_path=None,
#         table_size_array=None,
#         one_embedding_store_type="cached_host_mem",
#         cache_memory_budget_mb=8192,
#         use_bias=True
#     ):
#         super(LR, self).__init__()
#         self.bias = nn.Parameter(flow.tensor([1], dtype=flow.float32)) if use_bias else None
#         self.embedding_layer = OneEmbedding(
#             table_name="fm_lr_embedding",
#             embedding_vec_size=1,
#             persistent_path=persistent_path,
#             table_size_array=table_size_array,
#             store_type=one_embedding_store_type,
#             cache_memory_budget_mb=cache_memory_budget_mb
#         )

#     def forward(self, x):
#         # x = original ids
#         # order-1 feature interaction
#         embedded_x = self.embedding_layer(x)
#         output = flow.sum(embedded_x, dim=1)
#         if self.bias is not None:
#             output += self.bias
#         return output


# class Interaction(nn.Module):
#     def __init__(self, interaction_itself=False, num_fields=26):
#         super(Interaction, self).__init__()
#         self.interaction_itself = interaction_itself
#         self.num_fields = num_fields
        
#     def forward(self, embedded_x:flow.Tensor) -> flow.Tensor:
#         sum_of_square = flow.sum(embedded_x, dim=1) ** 2
#         square_of_sum = flow.sum(embedded_x ** 2, dim=1)
#         bi_interaction = (sum_of_square - square_of_sum) * 0.5
#         return flow.sum(bi_interaction, dim=-1).view(-1, 1)


# class FM(nn.Module):
#     def __init__(
#         self, 
#         persistent_path=None,
#         table_size_array=None,
#         one_embedding_store_type="cached_host_mem",
#         cache_memory_budget_mb=8192,
#         use_bias=True
#     ):
#         super(FM, self).__init__()
#         self.interaction = Interaction(num_fields=num_dense_fields+num_sparse_fields)
#         self.lr = LR(
#             persistent_path=persistent_path,
#             table_size_array=table_size_array,
#             one_embedding_store_type=one_embedding_store_type,
#             cache_memory_budget_mb=cache_memory_budget_mb,
#             use_bias=use_bias
#         )
    
#     def forward(self, x:flow.Tensor, embedded_x:flow.Tensor) -> flow.Tensor:
#         lr_out = self.lr(x)
#         dot_sum = self.interaction(embedded_x)
#         output = lr_out + dot_sum
#         return output


class InnerProductLayer(nn.Module):
    def __init__(self, interaction_type='dot', interaction_itself=False, num_sparse_fields=26):
        super(InnerProductLayer, self).__init__()
        self.interaction_type = interaction_type
        self.interaction_itself = interaction_itself
        self.num_sparse_fields = num_sparse_fields

        # slice
        offset = 1 if self.interaction_itself else 0
        # indices = flow.tensor([i * 27 + j for i in range(27) for j in range(i + offset)])
        # self.register_buffer("indices", indices)
        li = flow.tensor([i for i in range(39) for j in range(i + offset)])
        lj = flow.tensor([j for i in range(39) for j in range(i + offset)])
        self.register_buffer("li", li)
        self.register_buffer("lj", lj)
        

    # def forward(self, x:flow.Tensor) -> flow.Tensor:
    #     # x - dense fields, ly = embedding
    #     # if self.interaction_type == 'cat':
    #     #     R = flow.cat([x, ly], dim=1)
    #     if self.interaction_type == 'dot': # slice
    #         (batch_size, fields, d) = x.shape
    #         T = flow.cat([x], dim=1).view((batch_size, -1, d))
    #         # perform a dot product
    #         Z = flow.matmul(T, T, transpose_b=True)
    #         print("Z.shape: ", Z.shape)
    #         Zflat = Z[:, self.li, self.lj]
    #         print("Zflat.shape: ", Zflat.shape)
    #         R = flow.cat([Zflat], dim=1)       
    #     else:
    #         assert 0, 'dot or cat'
    #     return R

    def forward(self, x:flow.Tensor) -> flow.Tensor:
        # x - dense fields, ly = embedding
        # if self.interaction_type == 'cat':
        #     R = flow.cat([x, ly], dim=1)
        if self.interaction_type == 'dot': # slice
            # (batch_size, fields, d) = x.shape
            # T = flow.cat([x], dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = flow.matmul(x, x, transpose_b=True)
            print("Z.shape: ", Z.shape)
            Zflat = Z[:, self.li, self.lj]
            print("Zflat.shape: ", Zflat.shape)
            R = flow.cat([Zflat], dim=1)       
        else:
            assert 0, 'dot or cat'
        return R
    # def forward(self, x:flow.Tensor, ly:flow.Tensor) -> flow.Tensor:
    #     # x - dense fields, ly = embedding
    #     if self.interaction_type == 'cat':
    #         R = flow.cat([x, ly], dim=1)
    #     elif self.interaction_type == 'dot': # slice
    #         (batch_size, d) = x.shape
    #         T = flow.cat([x, ly], dim=1).view((batch_size, -1, d))
    #         # perform a dot product
    #         Z = flow.matmul(T, T, transpose_b=True)
    #         Zflat = Z[:, self.li, self.lj]
    #         R = flow.cat([x, Zflat], dim=1)       
    #     else:
    #         assert 0, 'dot or cat'
    #     return R

    # def output_feature_size(self, embedding_vec_size, dense_feature_size):
    #     if self.interaction_type == 'dot':
    #         # assert 0, 'dot is not supported yet'
    #         assert embedding_vec_size == dense_feature_size, "Embedding vector size must equle to dense feature size"
    #         n_cols = self.num_sparse_fields + 1
    #         if self.interaction_itself:
    #             n_cols += 1
    #         return dense_feature_size + sum(range(n_cols))
    #     elif self.interaction_type == 'cat':
    #         return embedding_vec_size * self.num_sparse_fields + dense_feature_size
    #     else:
    #         assert 0, 'dot or cat'


class PNNModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size=128,
        dnn=[1024, 1024, 512, 256],
        use_fusedmlp=True,
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        cache_memory_budget_mb=8192,
        interaction_type = 'dot',
        interaction_itself = False,
        dropout=0.2,
    ):
        super(PNNModule, self).__init__()

        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=embedding_vec_size,
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=3,
        )

        self.inner_product_layer = InnerProductLayer(interaction_type, interaction_itself, num_sparse_fields + num_sparse_fields)

        self.dnn_layer = DNN(
            in_features=embedding_vec_size * (num_dense_fields + num_sparse_fields) + sum(range(num_dense_fields + num_sparse_fields)),
            hidden_units=dnn+[1],
            skip_final_activation=True,
            fused=use_fusedmlp
        )

    def forward(self, inputs) -> flow.Tensor:
        T = self.embedding_layer(inputs)
        Z = self.inner_product_layer(T)
        print(Z.shape)
        dense_input = flow.cat([T.flatten(start_dim=1), Z], dim=1)
        print(dense_input.shape)
        dnn_pred = self.dnn_layer(dense_input)
        return dnn_pred


def make_pnn_module(args):
    model = PNNModule(
        embedding_vec_size=args.embedding_vec_size,
        dnn=args.dnn,
        use_fusedmlp=not args.disable_fusedmlp,
        persistent_path=args.persistent_path,
        table_size_array=args.table_size_array,
        one_embedding_store_type=args.store_type,
        cache_memory_budget_mb=args.cache_memory_budget_mb,
        dropout=args.net_dropout,
    )
    return model


class PNNValGraph(flow.nn.Graph):
    def __init__(self, pnn_module, amp=False):
        super(PNNValGraph, self).__init__()
        self.module = pnn_module
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        predicts = self.module(features.to("cuda"))
        return predicts.to("cpu")


class PNNTrainGraph(flow.nn.Graph):
    def __init__(
        self, pnn_module, loss, optimizer, lr_scheduler=None, grad_scaler=None, amp=False
    ):
        super(PNNTrainGraph, self).__init__()
        self.module = pnn_module
        self.loss = loss
        # self.max_norm = max_norm
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, features):
        logits = self.module(features.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        # TODO: add regularization for embedding
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")


def make_lr_scheduler(args, optimizer):
    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)
    milestones = [batches_per_epoch * (i + 1) for i in range(math.floor(math.log(args.min_lr / args.learning_rate, args.lr_factor)))]
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        gamma=args.lr_factor,
        milestones=milestones,
    )
    return multistep_lr


def early_stop(epoch, logs, best_metric, stopping_steps, patience=2, min_delta=1e-6):
    kv = {'auc': 1, 'logloss': -1}
    monitor_value = 0
    for k, v in kv.items():
        monitor_value += logs.get(k, 0) * v

    stop_training = False
    if monitor_value < best_metric + min_delta:
            stopping_steps += 1
            print("Monitor(max) STOP: {:.6f}!".format(monitor_value))
    else:
        stopping_steps = 0
        best_metric = monitor_value
    if stopping_steps >= patience:
        stop_training = True
        print(f"Early stopping at epoch={epoch}!")
    return stop_training, best_metric, stopping_steps


def train(args): 
    rank = flow.env.get_rank()

    pnn_module = make_pnn_module(args)
    pnn_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    if args.model_load_dir:
        print(f"Loading model from {args.model_load_dir}")
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        pnn_module.load_state_dict(state_dict, strict=False)

    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = pnn_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    # TODO: clip gradient norm
    # opt = flow.optim.SGD(pnn_module.parameters(), lr=args.learning_rate)
    # opt = flow.optim.SGD(
    #         [
    #             {
    #                 "params": pnn_module.parameters(),
    #                 "lr": args.learning_rate,
    #                 "clip_grad_max_norm": args.max_gradient_norm,
    #                 "clip_grad_norm_type": 2.0,
    #             }
    #         ],
    #     )

    opt = flow.optim.Adam(pnn_module.parameters(), lr=args.learning_rate)


    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )
    
    eval_graph = PNNValGraph(pnn_module, args.amp)
    train_graph = PNNTrainGraph(pnn_module, loss, opt, lr_scheduler, grad_scaler, args.amp)

    train_losses = []
    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)

    best_metric = -np.inf
    stopping_steps = 0

    pnn_module.train()
    step, last_step, last_time = -1, 0, time.time()
    epoch = 0
    with make_criteo_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        for step in range(1, args.train_batches + 1):
            labels, features = batch_to_global(*next(loader))
            loss = train_graph(labels, features)
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
                epoch += 1
                auc, logloss = eval(args, eval_graph, step, epoch)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{auc:0.5f}")

                stop_training, best_metric, stopping_steps = early_stop(
                    epoch, 
                    logs={'auc': auc, 'logloss': logloss}, 
                    best_metric=best_metric, 
                    stopping_steps=stopping_steps, 
                    patience=args.patience, 
                    min_delta=args.min_delta,
                )
                if stop_training:
                    break

                pnn_module.train()
                last_time = time.time()

    if step % batches_per_epoch != 0:
        auc = eval(args, eval_graph, step)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")
    
    plot_train_curve(args,train_losses)


def batch_to_global(np_label, np_features):
    def _np_to_global(np, dtype=flow.float):
        t = flow.tensor(np, dtype=dtype)
        return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
    labels = _np_to_global(np_label.reshape(-1, 1))
    features = _np_to_global(np_features, dtype=flow.int64)
    return labels, features


def eval(args, eval_graph, cur_step=0, epoch=0):
    if args.eval_batches <= 0:
        return
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    with make_criteo_dataloader(f"{args.data_dir}/test", args.eval_batch_size, shuffle=False) as loader:
        num_eval_batches = 0
        for np_batch in loader:
            num_eval_batches += 1
            if num_eval_batches > args.eval_batches:
                break
            label, features = batch_to_global(*np_batch)
            logits = eval_graph(features)
            pred = logits.sigmoid()
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
        log_loss_start_time = time.time()
        logloss = log_loss(labels, preds, eps=1e-7)
        log_loss_time = time.time() - log_loss_start_time
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Epoch {epoch}, Step {cur_step}, AUC {auc:0.5f}, LogLoss {logloss:0.5f}, Eval_time {eval_time:0.2f} s, "
            + f"AUC_time {auc_time:0.2f} s, LogLoss_time {log_loss_time: 0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    flow.comm.barrier()
    return auc, logloss


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
