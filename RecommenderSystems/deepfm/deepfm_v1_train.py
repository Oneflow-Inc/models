import argparse
import os
import sys
import glob
import time
import math
import numpy as np
import psutil
from typing import cast, Iterator, List, Optional, Tuple
import oneflow as flow
import oneflow.nn as nn
from petastorm.reader import make_batch_reader

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)


def get_args(argv:List[str],print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--num_train_samples",
        type=int,
        #required=True,
        default="128",
        help="the number of train samples",
    )
    parser.add_argument(
        "--num_val_samples",
        type=int,
        #required=True,
        default="128",
        help="the number of validation samples",
    )
    parser.add_argument(
        "--num_test_samples", type=int, 
        #required=True, 
        default="128",
        help="the number of test samples"
    )

    parser.add_argument(
        "--model_load_dir", type=str, default=None, help="model loading directory"
    )
    parser.add_argument(
        "--model_save_dir", type=str, default=None, help="model saving directory"
    )
    parser.add_argument(
        "--save_initial_model",
        action="store_true",
        help="save initial model parameters or not",
    )
    parser.add_argument(
        "--save_model_after_each_eval",
        action="store_true",
        help="save model after each eval or not",
    )

    parser.add_argument(
        "--embedding_vec_size", type=int, default=16, help="embedding vector size"
    )
    parser.add_argument(
        "--dnn",
        type=int_list,
        default="1000,1000,1000,1000,1000",
        help="dnn hidden units number",
    )
    parser.add_argument(
        "--net_dropout", type=float, default=0.2, help="net dropout rate"
    )
    parser.add_argument(
        "--disable_fusedmlp", action="store_true", help="disable fused MLP or not"
    )

    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )

    parser.add_argument(
        "--batch_size", type=int, default=10000, help="training/evaluation batch size"
    )
    parser.add_argument(
        "--train_batches",
        type=int,
        default=128,
        help="the maximum number of training batches",
    )
    parser.add_argument("--loss_print_interval",
                        type=int, default=100, help="")
    parser.add_argument(
        "--eval_batches", type=int, default=1612, help="number of eval batches"
    )
    parser.add_argument("--eval_batch_size", type=int, default=55296)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=27772)
    parser.add_argument("--decay_start", type=int, default=49315)

    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1.0e-6,
        help="threshold for measuring the new optimum, to only focus on significant changes",
    )

    parser.add_argument(
        "--table_size_array",
        type=int_list,
        help="embedding table size array for sparse fields",
        required=True,
    )
    parser.add_argument(
        "--persistent_path",
        type=str,
        required=True,
        help="path for persistent kv store",
    )
    parser.add_argument(
        "--store_type",
        type=str,
        default="cached_host_mem",
        help="OneEmbeddig persistent kv store type: device_mem, cached_host_mem, cached_ssd",
    )
    parser.add_argument(
        "--cache_memory_budget_mb",
        type=int,
        default=1024,
        help="size of cache memory budget on each device in megabytes when store_type is cached_host_mem or cached_ssd",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable Automatic Mixed Precision(AMP) training or not",
    )
    parser.add_argument(
        "--loss_scale_policy", type=str, default="static", help="static or dynamic"
    )

    parser.add_argument(
        "--disable_early_stop", action="store_true", help="enable early stop or not"
    )
    parser.add_argument(
        "--save_best_model", action="store_true", help="save best model or not"
    )

    # new add dataset_info num_dense_fields num_sparse_fields
    # 这个应该由样本决定
    parser.add_argument(
        "--num_dense_fields", type=int, default=13, help="the nums of dense fields for Feature"
    )

    parser.add_argument(
        "--num_sparse_fields", type=int, default=26, help="the nums of sparse fields for Feature"
    )
    args = parser.parse_args(argv)

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



class DLRMDataReader(object):
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
        num_dense_fields=13,
        num_sparse_fields=26
    ):
        self.parquet_file_url_list = parquet_file_url_list
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.shuffle_row_groups = shuffle_row_groups
        self.shard_seed = shard_seed
        self.shard_count = shard_count
        self.cur_shard = cur_shard

        fields = ["label"]
        fields += [f"I{i+1}" for i in range(num_dense_fields)]
        self.I_end = len(fields)
        fields += [f"C{i+1}" for i in range(num_sparse_fields)]
        self.C_end = len(fields)
        self.fields = fields

    def __enter__(self):
        self.reader = make_batch_reader(
            self.parquet_file_url_list,
            workers_count=1,
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
                        np.concatenate(
                            (tail[i], rglist[i][0 : (batch_size - len(tail[i]))])
                        )
                        for i in range(self.C_end)
                    ]
                )
                if len(tail[0]) == batch_size:
                    label = tail[0]
                    dense = tail[1 : self.I_end]
                    sparse = tail[self.I_end : self.C_end]
                    tail = None
                    yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                dense = [
                    rglist[j][pos : pos + batch_size] for j in range(1, self.I_end)
                ]
                sparse = [
                    rglist[j][pos : pos + batch_size]
                    for j in range(self.I_end, self.C_end)
                ]
                pos += batch_size
                yield label, np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.C_end)]


class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, relu=True) -> None:
        super(Dense, self).__init__()
        self.features = (
            nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(inplace=True))
            if relu
            else nn.Linear(in_features, out_features)
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.features(x)


class MLP(nn.Module):
    def __init__(
        self, in_features: int, hidden_units, skip_final_activation=False, fused=True
    ) -> None:
        super(MLP, self).__init__()
        if fused:
            self.linear_layers = nn.FusedMLP(
                in_features,
                hidden_units[:-1],
                hidden_units[-1],
                skip_final_activation=skip_final_activation,
            )
        else:
            units = [in_features] + hidden_units
            num_layers = len(hidden_units)
            denses = [
                Dense(
                    units[i],
                    units[i + 1],
                    not skip_final_activation or (i + 1) < num_layers,
                )
                for i in range(num_layers)
            ]
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, np.sqrt(2 / sum(param.shape)))
            elif "bias" in name:
                nn.init.normal_(param, 0.0, np.sqrt(1 / param.shape[0]))

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class Interaction(nn.Module):
    def __init__(
        self,
        dense_feature_size,
        num_embedding_fields,
        interaction_itself=False,
        interaction_padding=True,
    ):
        super(Interaction, self).__init__()
        self.interaction_itself = interaction_itself
        n_cols = (
            num_embedding_fields + 2
            if self.interaction_itself
            else num_embedding_fields + 1
        )
        output_size = dense_feature_size + sum(range(n_cols))
        self.output_size = (
            ((output_size + 8 - 1) // 8 * 8) if interaction_padding else output_size
        )
        self.output_padding = self.output_size - output_size

    def forward(self, x: flow.Tensor, y: flow.Tensor) -> flow.Tensor:
        (bsz, d) = x.shape
        return flow._C.fused_dot_feature_interaction(
            [x.view(bsz, 1, d), y],
            output_concat=x,
            self_interaction=self.interaction_itself,
            output_padding=self.output_padding,
        )

class OneEmbedding(nn.Module):
    """稀疏参数 oneEmbedding 构造类
    Args:
        embedding_vec_size (int): Embedding 的向量维度大小
        persistent_path (str): Embedding 的持久化路径
        table_size_array(List[int]):稀疏特征 id 统计数目 数组
        store_type(str): embedding 的存储介质 默认 device_mem:存储在设备当中,cached_host_mem:存储在主机内存当中,cached_ssd:存储在 ssd 硬盘当中
    """

    def __init__(
        self,
        embedding_vec_size: int,
        persistent_path: str,
        table_size_array: List[int],
        store_type: str = "device_mem",
        cache_memory_budget_mb: int = 1024*10,
        key_type: str = "int64",
        table_name:str= "sparse_embedding"
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)
        assert key_type in [
            "int32", "int64"], "OneEmbedding key_type must be integers"

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table_options(
                flow.one_embedding.make_uniform_initializer(
                    low=-scale, high=scale)
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
            table_name,
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=getattr(flow, key_type),
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class DNN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_units,
        out_features,
        skip_final_activation=False,
        dropout=0.0,
        fused=True,
    ) -> None:
        super(DNN, self).__init__()
        if fused:
            # y = ReLU(xA^T + b)
            self.dropout_rates = [dropout] * len(hidden_units)
            self.linear_layers = nn.FusedMLP(
                in_features,
                hidden_units,
                out_features,
                self.dropout_rates,
                0.0,
                skip_final_activation,
            )
        else:
            denses = []
            dropout_rates = [dropout] * len(hidden_units) + [0.0]
            use_relu = [True] * len(hidden_units) + [not skip_final_activation]
            hidden_units = [in_features] + hidden_units + [out_features]
            for idx in range(len(hidden_units) - 1):
                denses.append(
                    nn.Linear(hidden_units[idx],
                              hidden_units[idx + 1], bias=True)
                )
                if use_relu[idx]:
                    denses.append(nn.ReLU())
                if dropout_rates[idx] > 0:
                    denses.append(nn.Dropout(p=dropout_rates[idx]))
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


def interaction(embedded_x: flow.Tensor) -> flow.Tensor:
    return flow._C.fused_dot_feature_interaction([embedded_x], pooling="sum")


class DeepFMModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size=128,
        dnn=[1024, 1024, 512, 256],
        use_fusedmlp=True,
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        cache_memory_budget_mb=8192,
        dropout=0.2,
        num_dense_fields=13,
        num_sparse_fields=26,
        interaction_itself=True,
        interaction_padding=True,
        dense_input_padding=True,
        bottom_mlp=[512, 256, 128],
        top_mlp=[1024, 1024, 512, 256],
    ):
        super(DeepFMModule, self).__init__()

        self.embedding_vec_size = embedding_vec_size
        self.num_dense_fields = num_dense_fields
        self.num_sparse_fields = num_sparse_fields
        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=embedding_vec_size,
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb
        )
        # the deep for Deep&Wide 
        self.dnn_layer = DNN(
            in_features=embedding_vec_size *
            (1 + num_sparse_fields),
            hidden_units=dnn,
            out_features=1,
            skip_final_activation=True,
            dropout=dropout,
            fused=use_fusedmlp,
        )
        # the wide for Deep&Wide 
        
        #the dense feature pad 0
        self.num_dense_fields = (
            ((num_dense_fields + 8 - 1) // 8 * 8) if dense_input_padding else num_dense_fields
        )
        self.pad = (
            [0, self.num_dense_fields - num_dense_fields]
            if self.num_dense_fields > num_dense_fields
            else None
        )
        
        self.bottom_mlp = MLP(self.num_dense_fields, bottom_mlp, fused=use_fusedmlp)
        self.lr_out_mlp = MLP(self.num_sparse_fields*self.embedding_vec_size, [512,64,1], fused=True)
        
    #这里默认把 dense Feature 全部当做 稀疏参数了,后续需要基于 concat 的方式,否则 Embedding 会很大
    def forward2(self, inputs) -> flow.Tensor:
        multi_embedded_x = self.embedding_layer(inputs)
        embedded_x = multi_embedded_x[:, :, 0: self.embedding_vec_size]
        lr_embedded_x = multi_embedded_x[:, :, -1]

        # FM
        
        dot_sum = interaction(embedded_x)
        #fm_pred = lr_out + dot_sum

        # DNN
        dnn_pred = self.dnn_layer(embedded_x.flatten(start_dim=1))

        return  dnn_pred
    
    def forward(self, dense_fields, sparse_fields)->flow.Tensor:
        if self.pad:
            dense_fields = flow.nn.functional.pad(dense_fields, self.pad, "constant")
        dense_fields = flow.log(dense_fields + 1.0)
        # 稀疏 和 稠密
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding_layer(sparse_fields)
        # out = flow.cat([embedding, dense_fields], dim=1)
        
        # FM
        lr_in=embedding.flatten(start_dim=1)
        lr_out=self.lr_out_mlp(lr_in)
        dot_sum = interaction(embedding)
        fm_pred = lr_out + dot_sum
        
         # DNN
        dense_fields=dense_fields.unsqueeze(1)
        dnn_input = flow.cat([embedding, dense_fields], dim=1)
        dnn_input_flatten=dnn_input.flatten(start_dim=1)
        dnn_pred = self.dnn_layer(dnn_input_flatten)
        logit=fm_pred + dnn_pred
        return logit


def make_deepfm_module(args):
    model = DeepFMModule(
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

def make_criteo_dataloader(data_path, batch_size, shuffle=True):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return DLRMDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=1234,
        shard_count=world_size,
        cur_shard=flow.env.get_rank(),
    )


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=args.warmup_batches,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer,
        decay_batch=args.decay_batches,
        end_learning_rate=0,
        power=2.0,
        cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[args.decay_start],
        interval_rescaling=True,
    )
    return sequential_lr


class DLRMValGraph(flow.nn.Graph):
    def __init__(self, dlrm_module, amp=False):
        super(DLRMValGraph, self).__init__()
        self.module = dlrm_module
        if amp:
            self.config.enable_amp(True)

    def build(self, dense_fields, sparse_fields):
        predicts = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        return predicts.sigmoid()


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(
        self,
        dlrm_module,
        loss,
        optimizer,
        lr_scheduler=None,
        grad_scaler=None,
        amp=False,
    ):
        super(DLRMTrainGraph, self).__init__()
        self.module = dlrm_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, dense_fields, sparse_fields):
        logits = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()
        return reduce_loss.to("cpu")


def prefetch_eval_batches(data_dir, batch_size, num_batches):
    cached_eval_batches = []
    with make_criteo_dataloader(data_dir, batch_size, shuffle=False) as loader:
        for _ in range(num_batches):
            label, dense_fields, sparse_fields = batch_to_global(
                *next(loader), is_train=False
            )
            cached_eval_batches.append((label, dense_fields, sparse_fields))
    return cached_eval_batches



def train(args):
    rank = flow.env.get_rank()
    dlrm_module = make_deepfm_module(args)
    print(dlrm_module)
    # 基于 sbp 实现模型并行化, 广播Embedding 模型参数
    dlrm_module.to_global(
        flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
    #如果模型需要加载重新训练,加载模型
    if args.model_load_dir:
        print(f"Loading model from {args.model_load_dir}")
        state_dict = flow.load(args.model_load_dir, global_src_rank=0)
        dlrm_module.load_state_dict(state_dict, strict=False)

    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = dlrm_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")
    #设置模型网络优化器
    opt = flow.optim.SGD(dlrm_module.parameters(), lr=args.learning_rate)
    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")
    
    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
    eval_graph = DLRMValGraph(dlrm_module, args.amp)
    train_graph = DLRMTrainGraph(
        dlrm_module, loss, opt, lr_scheduler, grad_scaler, args.amp
    )

    cached_eval_batches = prefetch_eval_batches(
        f"{args.data_dir}/test", args.eval_batch_size, args.eval_batches
    )

    dlrm_module.train()
    max_auc=0.0
    with make_criteo_dataloader(
        f"{args.data_dir}/train", args.batch_size
    ) as loader:
        step, last_step, last_time = -1, 0, time.time()
        for step in range(1, args.train_batches + 1):
            labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            loss = train_graph(labels, dense_fields, sparse_fields)
            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, Latency "
                        + f"{(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            if args.eval_interval > 0 and step % args.eval_interval == 0:
                auc = eval(cached_eval_batches, eval_graph, step)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{auc:0.5f}")
                dlrm_module.train()
                last_time = time.time()

    if args.eval_interval > 0 and step % args.eval_interval != 0:
        auc = eval(cached_eval_batches, eval_graph, step)
        if auc > max_auc:
            auc = max_auc
        if args.save_model_after_each_eval:
            print(f"save the  model step:{step} auc:{auc} max_auc:{max_auc} ")
            save_model(f"step_{step}_val_auc_{auc:0.5f}")
            
    for name, param in dlrm_module.named_parameters(recurse=True):
        #print(f"name:{name} param:{param}")
        print(f"name:{name} param-size:{param.size()}")
        
def np_to_global(np):
    t = flow.from_numpy(np)
    return t.to_global(
        placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
    )


def batch_to_global(np_label, np_dense, np_sparse, is_train=True):
    
    labels=np_label.astype(np.float32)
    labels = (
        np_to_global(np_label.reshape(-1, 1)) if is_train else np_label.reshape(-1, 1)
    )
    dense_fields = np_to_global(np_dense)
    sparse_fields = np_to_global(np_sparse)
    return labels, dense_fields, sparse_fields


def eval(cached_eval_batches, eval_graph, cur_step=0):
    num_eval_batches = len(cached_eval_batches)
    if num_eval_batches <= 0:
        return
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    for i in range(num_eval_batches):
        label, dense_fields, sparse_fields = cached_eval_batches[i]
        pred = eval_graph(dense_fields, sparse_fields)
        labels.append(label)
        preds.append(pred.to_local())

    labels = (
        np_to_global(np.concatenate(labels, axis=0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )
    preds = (
        flow.cat(preds, dim=0)
        .to_global(
            placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
        )
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )
    flow.comm.barrier()
    eval_time = time.time() - eval_start_time

    rank = flow.env.get_rank()
    auc = 0
    if rank == 0:
        auc_start_time = time.time()
        auc = flow.roc_auc_score(labels, preds).numpy()[0]
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

    return auc

    

if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    
    debug = True
    if debug:
        argv=[
            "--data_dir" ,"/home/zhipeng.li/data/criteo_spark/parquet",
            "--persistent_path", "/home/zhipeng.li/tmp/oneflow/deepfm/persistent",
            #--table_size_array "62866, 8001, 2901, 74623, 7530, 3391, 1400, 21705, 7937, 21, 276, 1235896, 9659, 39884407, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532952, 2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979772, 25641295, 39664985, 585935, 12972, 108, 36" \
            "--table_size_array", "39884407, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532952, 2953546, 403346, 10, 2208, 11938, 155, 4, 976, 14, 39979772, 25641295, 39664985, 585935, 12972, 108, 36" ,
            "--store_type", "cached_host_mem" ,
            "--cache_memory_budget_mb", "1024",
            "--batch_size", "64",
            "--num_train_samples", "6400",
            "--num_val_samples", "6400",
            "--num_test_samples","6400",
            "--train_batches", "100",
            "--eval_batches", "10",
            "--eval_batch_size", "64",
            "--eval_interval","1",
            "--warmup_batches","10",
            "--decay_batches","10",
            "--decay_start","10",
            "--loss_print_interval", "1",
            "--dnn", "1000,1000,1000,1000,1000",
            "--net_dropout", "0.2" ,
            "--learning_rate", "0.001",
            "--embedding_vec_size", "128",
            "--model_save_dir","/home/zhipeng.li/tmp/oneflow/deepfm/model/save/dir",
            "--save_best_model"  
        ]
        #main(sys.argv[1:])
        args = get_args(argv)
        train(args)
    else:
        args = get_args(sys.argv[1:])
        train(args)
