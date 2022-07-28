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


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--num_train_samples", type=int, required=True, help="the number of train samples"
    )
    parser.add_argument(
        "--num_test_samples", type=int, required=True, help="the number of test samples"
    )

    parser.add_argument("--model_load_dir", type=str, default=None, help="model loading directory")
    parser.add_argument("--model_save_dir", type=str, default=None, help="model saving directory")
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not"
    )
    parser.add_argument(
        "--save_model_after_each_eval",
        action="store_true",
        help="save model after each eval or not",
    )

    parser.add_argument("--embedding_vec_size", type=int, default=16, help="embedding vector size")
    parser.add_argument(
        "--dnn", type=int_list, default="1000,1000,1000,1000,1000", help="dnn hidden units number"
    )
    parser.add_argument("--net_dropout", type=float, default=0.2, help="net dropout rate")
    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")

    parser.add_argument(
        "--batch_size", type=int, default=10000, help="training/evaluation batch size"
    )
    parser.add_argument(
        "--train_batches", type=int, default=75000, help="the maximum number of training batches"
    )
    parser.add_argument("--loss_print_interval", type=int, default=100, help="")
    parser.add_argument("--eval_interval", type=int, default=100000)

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
        "--persistent_path", type=str, required=True, help="path for persistent kv store"
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
        "--amp", action="store_true", help="enable Automatic Mixed Precision(AMP) training or not"
    )
    parser.add_argument("--loss_scale_policy", type=str, default="static", help="static or dynamic")

    parser.add_argument(
        "--disable_early_stop", action="store_true", help="enable early stop or not"
    )
    parser.add_argument("--save_best_model", action="store_true", help="save best model or not")

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
            flow.one_embedding.make_table_options(
                [
                    flow.one_embedding.make_column_options(
                        flow.one_embedding.make_normal_initializer(mean=0, std=1e-4)
                    ),
                    flow.one_embedding.make_column_options(
                        flow.one_embedding.make_normal_initializer(mean=0, std=1e-4)
                    ),
                ]
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
        self.one_embedding = flow.one_embedding.MultiTableMultiColumnEmbedding(
            name=table_name,
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int32,
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
                denses.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=True))
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
    ):
        super(DeepFMModule, self).__init__()

        self.embedding_vec_size = embedding_vec_size

        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=[embedding_vec_size, 1],
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=3,
        )

        self.dnn_layer = DNN(
            in_features=embedding_vec_size * (num_dense_fields + num_sparse_fields),
            hidden_units=dnn,
            out_features=1,
            skip_final_activation=True,
            dropout=dropout,
            fused=use_fusedmlp,
        )

    def forward(self, inputs) -> flow.Tensor:
        multi_embedded_x = self.embedding_layer(inputs)
        embedded_x = multi_embedded_x[:, :, 0 : self.embedding_vec_size]
        lr_embedded_x = multi_embedded_x[:, :, -1]
        
        # FM
        lr_out = flow.sum(lr_embedded_x, dim=1, keepdim=True)
        dot_sum = interaction(embedded_x)
        fm_pred = lr_out + dot_sum

        # DNN
        dnn_pred = self.dnn_layer(embedded_x.flatten(start_dim=1))

        return fm_pred + dnn_pred


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


class DeepFMValGraph(flow.nn.Graph):
    def __init__(self, deepfm_module, label_loader, sparse_loader, amp=False):
        super(DeepFMValGraph, self).__init__()
        self.module = deepfm_module
        self.label_loader = label_loader
        self.sparse_loader = sparse_loader
        if amp:
            self.config.enable_amp(True)

    def build(self):
        labels = self.label_loader()
        sparse_fields = self.sparse_loader()
        predicts = self.module(sparse_fields.to("cuda"))
        return labels, predicts.sigmoid()


class DeepFMTrainGraph(flow.nn.Graph):
    def __init__(
        self, deepfm_module, loss, optimizer, label_loader, sparse_loader, grad_scaler=None, amp=False, lr_scheduler=None,
    ):
        super(DeepFMTrainGraph, self).__init__()
        self.module = deepfm_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        self.label_loader = label_loader
        self.sparse_loader = sparse_loader
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


def make_lr_scheduler(args, optimizer):
    warmup_lr = flow.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0, total_iters=3000,
    )
    poly_decay_lr = flow.optim.lr_scheduler.PolynomialLR(
        optimizer, decay_batch=60000, end_learning_rate=1e-8, power=2.0, cycle=False,
    )
    sequential_lr = flow.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_lr, poly_decay_lr],
        milestones=[10000],
        interval_rescaling=True,
    )
    return sequential_lr


def train(args):
    rank = flow.env.get_rank()

    deepfm_module = make_deepfm_module(args)
    deepfm_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)
    label_loader = flow.nn.RawReader(
            ["/RAID0/criteo1t_oneflow_raw/train/label_float.bin"],
            (1,),
            flow.float32,
            args.batch_size,
            random_shuffle=True,
            random_seed=1234,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0)
        )

    sparse_loader = flow.nn.RawReader(
            ["/RAID0/criteo1t_oneflow_raw/train/sparse_C39_int32.bin"],
            (39,),
            flow.int32,
            args.batch_size,
            random_shuffle=True,
            random_seed=1234,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0)
        )

    label_loader_val = flow.nn.RawReader(
            ["/RAID0/criteo1t_oneflow_raw/test/label_float.bin"],
            (1,),
            flow.float32,
            args.batch_size,
            random_shuffle=False,
            random_seed=1234,
            placement=flow.env.all_device_placement("cpu"),
            sbp=flow.sbp.split(0)
        )

    sparse_loader_val = flow.nn.RawReader(
            ["/RAID0/criteo1t_oneflow_raw/test/sparse_C39_int32.bin"],
            (39,),
            flow.int32,
            args.batch_size,
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
            deepfm_module.load_state_dict(state_dict, strict=False)
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
        state_dict = deepfm_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    # TODO: clip gradient norm
    opt = flow.optim.Adam(deepfm_module.parameters(), lr=args.learning_rate, eps=1e-7)
    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="mean").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DeepFMValGraph(deepfm_module, label_loader_val, sparse_loader_val, args.amp)
    train_graph = DeepFMTrainGraph(deepfm_module, loss, opt, label_loader, sparse_loader, grad_scaler, args.amp, lr_scheduler=lr_scheduler)

    best_metric = -np.inf
    stopping_steps = 0
    save_best = False
    stop_training = False

    deepfm_module.train()
    if True:
        step, last_step, last_time = -1, 0, time.time()
        for step in range(1, args.train_batches + 1):
            loss = train_graph()
            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.batch_size / latency
                    last_step, last_time = step, time.time()
                    #strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    strtime = time.time()
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            if (args.eval_interval > 0 and step % args.eval_interval == 0):
                auc, logloss = eval(
                    args,
                    eval_graph,
                    cur_step=step,
                )
                # if args.save_model_after_each_eval:
                #     save_model(f"step_{step}_val_auc_{auc:0.5f}")

                # if args.save_best_model and save_best:
                #     if rank == 0:
                #         print(f"Save best model: monitor(max): {best_metric:.6f}")
                #     save_model("best_checkpoint")

                # if not args.disable_early_stop and stop_training:
                #     break

                deepfm_module.train()
                last_time = time.time()

    if args.save_best_model:
        load_model(f"{args.model_save_dir}/best_checkpoint")
    if rank == 0:
        print("================ Test Evaluation ================")
    if (args.eval_interval > 0 and step % args.eval_interval != 0):
        eval(args, eval_graph, cur_step=step)


def eval(args, eval_graph, cur_step=0):
    batches_per_epoch = math.ceil(args.num_test_samples / args.batch_size)

    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()

    for i in range(batches_per_epoch):
        label, pred = eval_graph()
        labels.append(label.to_local())
        preds.append(pred.to_local())

    labels = (
        flow.cat(labels, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )
    preds = (
        flow.cat(preds, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )

    flow.comm.barrier()
    eval_time = time.time() - eval_start_time

    rank = flow.env.get_rank()

    metrics_start_time = time.time()
    auc = flow.roc_auc_score(labels, preds).numpy()[0]
    #logloss = flow._C.binary_cross_entropy_loss(preds, labels, weight=None, reduction="mean")
    logloss = 0
    metrics_time = time.time() - metrics_start_time

    if rank == 0:
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Step {cur_step}, AUC {auc:0.6f}, LogLoss {logloss:0.6f}, "
            + f"Eval_time {eval_time:0.2f} s, Metrics_time {metrics_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    return auc, logloss


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)

