import argparse
import os
import sys
import glob
import time
import math
import numpy as np
import psutil
import oneflow as flow
import oneflow.nn as nn
from petastorm.reader import make_batch_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
num_dense_fields = 13
num_sparse_fields = 26
num_fields = num_dense_fields + num_sparse_fields


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--num_train_samples", type=int, default=4195197692, help="the number of training samples",
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=89137319, help="the number of test samples",
    )

    parser.add_argument("--shard_seed", type=int, default=2022)
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not.",
    )
    parser.add_argument(
        "--save_model_after_each_eval", action="store_true", help="save model after each eval.",
    )

    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=16)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--dnn_hidden_units", type=int_list, default="1000,1000,1000,1000,1000")
    parser.add_argument("--crossing_layers", type=int, default=4)
    parser.add_argument("--net_dropout", type=float, default=0.05)

    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--size_factor", type=int, default=3)

    parser.add_argument("--test_batch_size", type=int, default=55296)
    parser.add_argument("--test_batches", type=int, default=1612, help="number of test batches")
    parser.add_argument("--train_batch_size", type=int, default=55296)
    parser.add_argument("--train_batches", type=int, default=75869, help="number of train batches")
    parser.add_argument("--loss_print_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100000)

    parser.add_argument(
        "--table_size_array",
        type=int_list,
        help="Embedding table size array for sparse fields",
        default="62866,8001,2901,74623,7530,3391,1400,21705,7937,21,276,1235896,9659,39884301,39040,17291,7421,20263,3,7121,1543,63,38532372,2953790,403302,10,2209,11938,155,4,976,14,39979538,25638302,39665755,585840,12973,108,36",
    )
    parser.add_argument(
        "--persistent_path", type=str, default="persistent", help="path for persistent kv store",
    )
    parser.add_argument("--store_type", type=str, default="device_mem")
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

        tables = [
            flow.one_embedding.make_table(
                flow.one_embedding.make_normal_initializer(mean=0.0, std=1e-4)
            )
            for _ in range(len(table_size_array))
        ]

        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, capacity=vocab_size, size_factor=size_factor,
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
            key_type=flow.int32,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.add_parameters()
        self.reset_parameters()

    def add_parameters(self) -> None:
        for idx in range(self.num_layers):
            self.register_parameter(
                f"weight_{idx}", flow.nn.Parameter(flow.Tensor(1, self.input_dim,)),
            )
            self.register_parameter(
                f"bias_{idx}", flow.nn.Parameter(flow.zeros(self.input_dim)),
            )

    def weight(self, i):
        return getattr(self, f"weight_{i}")

    def bias(self, i):
        return getattr(self, f"bias_{i}")

    def reset_parameters(self) -> None:
        for i in range(self.num_layers):
            flow.nn.init.kaiming_uniform_(self.weight(i), a=math.sqrt(5))

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = flow._C.fused_cross_feature_interaction(
                X_i, self.weight(i), X_0, self.bias(i), "vector"
            )
        return X_i


class DNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=[],
        dropout_rates=0,
        use_fusedmlp=True,
        batch_norm=False,
        use_bias=True,
    ):
        super(DNN, self).__init__()
        dense_layers = []
        if use_fusedmlp and not batch_norm:
            hidden_dropout_rates_list = [dropout_rates] * (len(hidden_units) - 1)
            self.dnn = nn.FusedMLP(
                input_dim,
                hidden_units[:-1],
                hidden_units[-1],
                hidden_dropout_rates_list,
                dropout_rates,
                False,
            )
        else:
            hidden_units = [input_dim] + hidden_units
            for idx in range(len(hidden_units) - 1):
                dense_layers.append(
                    nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)
                )
                dense_layers.append(nn.ReLU())
                if batch_norm:
                    dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
                if dropout_rates > 0:
                    dense_layers.append(nn.Dropout(p=dropout_rates))
            self.dnn = nn.Sequential(*dense_layers)  # * used to unpack list

    def forward(self, inputs):
        return self.dnn(inputs)


class DCNModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        one_embedding_store_type,
        cache_memory_budget_mb,
        size_factor,
        dnn_hidden_units=[128, 128],
        use_fusedmlp=True,
        crossing_layers=3,
        net_dropout=0.2,
        batch_norm=False,
    ):
        super(DCNModule, self).__init__()

        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=embedding_vec_size,
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=size_factor,
        )

        input_dim = embedding_vec_size * num_fields

        self.dnn = (
            DNN(
                input_dim=input_dim,
                hidden_units=dnn_hidden_units,
                dropout_rates=net_dropout,
                use_fusedmlp=use_fusedmlp,
                batch_norm=batch_norm,
                use_bias=True,
            )
            if dnn_hidden_units
            else None
        )  # in case of only crossing net used

        self.crossnet = CrossNet(input_dim, crossing_layers)

        final_dim = input_dim
        if isinstance(dnn_hidden_units, list) and len(dnn_hidden_units) > 0:  # if use dnn
            final_dim += dnn_hidden_units[-1]
        self.fc = nn.Linear(final_dim, 1)  # [cross_part, dnn_part] -> logit

        self.reset_parameters()

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
        dnn_hidden_units=args.dnn_hidden_units,
        use_fusedmlp=not args.disable_fusedmlp,
        crossing_layers=args.crossing_layers,
        net_dropout=args.net_dropout,
        batch_norm=args.batch_norm,
        size_factor=args.size_factor,
    )
    return model


class DCNValGraph(flow.nn.Graph):
    def __init__(self, dcn_module, data_dir, batch_size, amp=False):
        super(DCNValGraph, self).__init__()
        self.module = dcn_module
        self.label_loader, self.sparse_loader = make_raw_dataloader(
            f"{data_dir}/test", batch_size, shuffle=False
        )
        if amp:
            self.config.enable_amp(True)

    def build(self):
        labels = self.label_loader()
        sparse_fields = self.sparse_loader()
        predicts = self.module(sparse_fields.to("cuda"))
        return labels, predicts.sigmoid()


class DCNTrainGraph(flow.nn.Graph):
    def __init__(
        self, dcn_module, data_dir, batch_size, loss, optimizer, grad_scaler=None, amp=False, lr_scheduler=None, 
    ):
        super(DCNTrainGraph, self).__init__()
        self.module = dcn_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        self.label_loader, self.sparse_loader = make_raw_dataloader(
            f"{data_dir}/train", batch_size
        )
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self):
        labels = self.label_loader()
        sparse_fields = self.sparse_loader()
        logits = self.module(sparse_fields.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        loss.backward()
        return loss.to("cpu")


def make_raw_dataloader(data_path, batch_size, shuffle=True):
    def make_reader(data_file, length, dtype):
        return flow.nn.RawReader(
            [data_file],
            (length,),
            dtype,
            batch_size,
            random_shuffle=shuffle,
            random_seed=1234,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0),
        )

    label_loader = make_reader(f"{data_path}/label.bin", 1, flow.float32)
    sparse_loader = make_reader(f"{data_path}/sparse_C39.bin", num_fields, flow.int32)

    return label_loader, sparse_loader


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=2750,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, decay_batch=40000, end_learning_rate=1e-6, power=2.0, cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[40000],
        interval_rescaling=True,
    )
    return sequential_lr


def train(args):
    rank = flow.env.get_rank()

    dcn_module = make_dcn_module(args)
    dcn_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    sparse_loader_val = flow.nn.RawReader(
            ["/RAID0/criteo1t_oneflow_raw/test/sparse_C39_int32.bin"],
            (39,),
            flow.int32,
            args.test_batch_size,
            random_shuffle=False,
            random_seed=1234,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0)
        )

    def load_model(dir):
        if rank == 0:
            print(f"Loading model from {dir}")
        if os.path.exists(dir):
            state_dict = flow.load(dir, global_src_rank=0)
            dcn_module.load_state_dict(state_dict, strict=False)
        else:
            if rank == 0:
                print(f"Loading model from {dir} failed: invalid path")

    if args.model_load_dir:
        load_model(args.model_load_dir)

    def save_model(subdir):
        if not args.model_save_dir:
            return
        save_path = os.path.join(args.model_save_dir, subdir)
        if rank == 0:
            print(f"Saving model to {save_path}")
        state_dict = dcn_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    opt = flow.optim.Adam(dcn_module.parameters(), lr=args.learning_rate)
    lr_scheduler = None
    # loss_func = flow.nn.BCELoss(reduction="none").to("cuda")
    loss_func = flow.nn.BCEWithLogitsLoss(reduction="mean").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DCNValGraph(dcn_module, args.data_dir, args.test_batch_size, args.amp)
    train_graph = DCNTrainGraph(dcn_module, args.data_dir, args.train_batch_size, loss_func, opt, grad_scaler, args.amp, lr_scheduler=lr_scheduler)

    best_metric = -np.inf
    stopping_steps = 0
    stop_training = False

    dcn_module.train()
    if True:
        last_step, last_time = 0, time.time()
        for step in range(1, args.train_batches + 1):
            loss = train_graph()

            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.train_batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    strtime = time.time()
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            if (args.eval_interval > 0 and step % args.eval_interval == 0):
                val_auc = eval(args, eval_graph, cur_step=step)
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{val_auc:0.5f}")

                dcn_module.train()
                last_time = time.time()

    if rank == 0:
        print("================ Test Evaluation ================")
    if (args.eval_interval > 0 and step % args.eval_interval != 0):
        eval(args, eval_graph, cur_step=step)


def tensor_list_to_local(tensors):
    return (
        flow.cat(tensors, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )


def eval(args, eval_graph, cur_step=0):
    batches_per_epoch = math.ceil(args.num_test_samples / args.test_batch_size)
    batch_size = args.test_batch_size

    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()

    for i in range(batches_per_epoch):
        label, pred = eval_graph()
        labels.append(label.to_local())
        preds.append(pred.to_local())

    labels = tensor_list_to_local(labels)
    preds = tensor_list_to_local(preds)

    flow.comm.barrier()
    eval_time = time.time() - eval_start_time

    rank = flow.env.get_rank()

    metrics_start_time = time.time()
    auc = flow.roc_auc_score(labels, preds).numpy()[0]
    metrics_time = time.time() - metrics_start_time

    if rank == 0:
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Step {cur_step}, AUC {auc:0.6f}, "
            + f"Eval_time {eval_time:0.2f} s, Metrics_time {metrics_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    return auc


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)
