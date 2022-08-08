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
        "--num_train_samples", type=int, default=36672493, help="the number of training samples"
    )
    parser.add_argument(
        "--num_valid_samples", type=int, default=4584062, help="the number of validation samples"
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=4584062, help="the number of test samples"
    )

    parser.add_argument("--shard_seed", type=int, default=2022)
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument("--save_best_model", action="store_true", help="save best model or not")
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not."
    )
    parser.add_argument(
        "--save_model_after_each_eval", action="store_true", help="save model after each eval."
    )
    parser.add_argument("--embedding_vec_size", type=int, default=16)
    parser.add_argument("--batch_norm", type=bool, default=False)
    parser.add_argument("--dnn_hidden_units", type=int_list, default="1000,1000,1000,1000,1000")
    parser.add_argument("--crossing_layers", type=int, default=3)
    parser.add_argument("--net_dropout", type=float, default=0.2)
    parser.add_argument("--embedding_regularizer", type=float, default=None)
    parser.add_argument("--net_regularizer", type=float, default=None)

    parser.add_argument(
        "--disable_early_stop", action="store_true", help="enable early stop or not"
    )
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min_delta", type=float, default=1.0e-6)
    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--size_factor", type=int, default=3)

    parser.add_argument("--valid_batch_size", type=int, default=10000)
    parser.add_argument("--valid_batches", type=int, default=1000, help="number of valid batches")
    parser.add_argument("--test_batch_size", type=int, default=10000)
    parser.add_argument("--test_batches", type=int, default=1000, help="number of test batches")
    parser.add_argument("--train_batch_size", type=int, default=10000)
    parser.add_argument("--train_batches", type=int, default=15000, help="number of train batches")
    parser.add_argument("--loss_print_interval", type=int, default=100)

    parser.add_argument(
        "--table_size_array",
        type=int_list,
        help="Embedding table size array for sparse fields",
        required=True,
    )
    parser.add_argument(
        "--persistent_path", type=str, required=True, help="path for persistent kv store"
    )
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


class DCNDataReader(object):
    """A context manager that manages the creation and termination of a
    :class:`petastorm.Reader`.
    """

    def __init__(
        self,
        parquet_file_url_list,
        batch_size,
        num_epochs=1,
        shuffle_row_groups=True,
        shard_seed=2019,
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
                    features = tail[1 : self.num_fields]
                    tail = None
                    yield label, np.stack(features, axis=-1)
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                label = rglist[0][pos : pos + batch_size]
                features = [rglist[j][pos : pos + batch_size] for j in range(1, self.num_fields)]
                pos += batch_size
                yield label, np.stack(features, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.num_fields)]


def make_criteo_dataloader(data_path, batch_size, shuffle=True, shard_seed=2022):
    """Make a Criteo Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return DCNDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=shard_seed,
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

        tables = [
            flow.one_embedding.make_table_options(
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
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class CrossInteractionLayer(nn.Module):
    '''
    Follow the same CrossInteractionLayer implementation of FuxiCTR
    '''
    def __init__(self, input_dim):
        super(CrossInteractionLayer, self).__init__()
        self.weight = nn.Linear(input_dim, 1, bias=False)
        self.bias = nn.Parameter(flow.zeros(input_dim))

    def forward(self, X_0, X_i):
        interaction_out = self.weight(X_i) * X_0 + self.bias
        return interaction_out


class CrossNet(nn.Module):
    '''
    Follow the same CrossNet implementation of FuxiCTR
    '''
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layers = num_layers
        self.cross_net = nn.ModuleList(
            CrossInteractionLayer(input_dim) for _ in range(self.num_layers)
        )

    def forward(self, X_0):
        X_i = X_0  # b x dim
        for i in range(self.num_layers):
            X_i = X_i + self.cross_net[i](X_0, X_i)
        return X_i


class DNN(nn.Module):
    def __init__(
        self, input_dim, hidden_units=[], dropout_rates=0, batch_norm=False, use_bias=True,
    ):
        super(DNN, self).__init__()
        dense_layers = []
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
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

        input_dim = embedding_vec_size * (num_dense_fields + num_sparse_fields)

        self.dnn = (
            DNN(
                input_dim=input_dim,
                hidden_units=dnn_hidden_units,
                dropout_rates=net_dropout,
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
        return y_pred.sigmoid()

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
        crossing_layers=args.crossing_layers,
        net_dropout=args.net_dropout,
        batch_norm=args.batch_norm,
        size_factor=args.size_factor,
    )
    return model


class DCNValGraph(flow.nn.Graph):
    def __init__(self, dcn_module, amp=False):
        super(DCNValGraph, self).__init__()
        self.module = dcn_module
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        predicts = self.module(features.to("cuda"))
        return predicts


class DCNTrainGraph(flow.nn.Graph):
    def __init__(
        self, dcn_module, loss, optimizer, lr_scheduler=None, grad_scaler=None, amp=False,
    ):
        super(DCNTrainGraph, self).__init__()
        self.module = dcn_module
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
        reduce_loss = flow.mean(loss)
        reduce_loss.backward()

        return reduce_loss.to("cpu")


def make_lr_scheduler(args, optimizer):
    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[3 * batches_per_epoch], gamma=args.lr_factor
    )
    return multistep_lr


def train(args):
    rank = flow.env.get_rank()
    dcn_module = make_dcn_module(args)
    dcn_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

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

    def get_metrics(logs):
        kv = {"auc": 1, "logloss": -1}
        monitor_value = 0
        for k, v in kv.items():
            monitor_value += logs.get(k, 0) * v
        return monitor_value

    def early_stop(
        epoch, monitor_value, best_metric, stopping_steps, patience=2, min_delta=1e-6,
    ):
        rank = flow.env.get_rank()
        save_best = False
        stop_training = False
        if monitor_value < best_metric + min_delta:
            stopping_steps += 1
            if rank == 0:
                print("Monitor(max) STOP: {:.6f}!".format(monitor_value))
        else:
            stopping_steps = 0
            best_metric = monitor_value
            save_best = True
        if stopping_steps >= patience:
            stop_training = True
            if rank == 0:
                print(f"Early stopping at epoch={epoch}!")
        return stop_training, best_metric, stopping_steps, save_best

    opt = flow.optim.Adam(dcn_module.parameters(), lr=args.learning_rate)
    lr_scheduler = None
    loss_func = flow.nn.BCELoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DCNValGraph(dcn_module, args.amp)
    train_graph = DCNTrainGraph(dcn_module, loss_func, opt, lr_scheduler, grad_scaler, args.amp)

    batches_per_epoch = math.ceil(args.num_train_samples / args.train_batch_size)

    best_metric = -np.inf
    epoch = 0
    stopping_steps = 0
    stop_training = False

    cached_valid_batches = prefetch_eval_batches(
        f"{args.data_dir}/val", args.valid_batch_size, args.valid_batches
    )

    with make_criteo_dataloader(
        f"{args.data_dir}/train", args.train_batch_size, shard_seed=args.shard_seed
    ) as loader:
        dcn_module.train()
        last_step, last_time = 0, time.time()
        for step in range(1, args.train_batches + 1):
            labels, features = batch_to_global(*next(loader))
            loss = train_graph(labels, features)

            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.train_batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            if step % batches_per_epoch == 0:
                epoch += 1
                val_auc, val_logloss = eval(
                    args,
                    eval_graph,
                    tag="val",
                    cur_step=step,
                    epoch=epoch,
                    cached_eval_batches=cached_valid_batches,
                )
                if args.save_model_after_each_eval:
                    save_model(f"step_{step}_val_auc_{val_auc:0.5f}")

                monitor_value = get_metrics(logs={"auc": val_auc, "logloss": val_logloss})

                stop_training, best_metric, stopping_steps, save_best = early_stop(
                    epoch,
                    monitor_value,
                    best_metric=best_metric,
                    stopping_steps=stopping_steps,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )

                if args.save_best_model and save_best:
                    if rank == 0:
                        print(f"Save best model: monitor(max): {best_metric:.6f}")
                    save_model("best_checkpoint")

                if not args.disable_early_stop and stop_training:
                    break

                dcn_module.train()
                last_time = time.time()
    if args.save_best_model:
        load_model(f"{args.model_save_dir}/best_checkpoint")
    if rank == 0:
        print("================ Test Evaluation ================")
    eval(args, eval_graph, tag="test", cur_step=step, epoch=epoch)


def _np_to_global(np_array):
    t = flow.from_numpy(np_array)
    return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))


def batch_to_global(np_label, np_features, is_train=True):
    labels = _np_to_global(np_label.reshape(-1, 1)) if is_train else np_label.reshape(-1, 1)
    features = _np_to_global(np_features)
    return labels, features


def prefetch_eval_batches(data_dir, batch_size, num_batches):
    cached_eval_batches = []
    with make_criteo_dataloader(data_dir, batch_size, shuffle=False) as loader:
        for _ in range(num_batches):
            label, features = batch_to_global(*next(loader), is_train=False)
            cached_eval_batches.append((label, features))
    return cached_eval_batches


def eval(args, eval_graph, tag="val", cur_step=0, epoch=0, cached_eval_batches=None):
    if tag == "val":
        batches_per_epoch = math.ceil(args.num_valid_samples / args.valid_batch_size)
        batch_size = args.valid_batch_size
    else:
        batches_per_epoch = math.ceil(args.num_test_samples / args.test_batch_size)
        batch_size = args.test_batch_size

    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()

    if cached_eval_batches == None:
        with make_criteo_dataloader(f"{args.data_dir}/{tag}", batch_size, shuffle=False) as loader:
            eval_start_time = time.time()
            for i in range(batches_per_epoch):
                label, features = batch_to_global(*next(loader), is_train=False)
                pred = eval_graph(features)
                labels.append(label)
                preds.append(pred.to_local())
    else:
        for i in range(batches_per_epoch):
            label, features = cached_eval_batches[i]
            pred = eval_graph(features)
            labels.append(label)
            preds.append(pred.to_local())

    labels = (
        _np_to_global(np.concatenate(labels, axis=0)).to_global(sbp=flow.sbp.broadcast()).to_local()
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
    logloss = flow._C.binary_cross_entropy_loss(preds, labels, weight=None, reduction="mean").item()
    metrics_time = time.time() - metrics_start_time

    if rank == 0:
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"Rank[{rank}], Epoch {epoch}, Step {cur_step}, AUC {auc:0.6f}, LogLoss {logloss:0.6f}, "
            + f"Eval_time {eval_time:0.2f} s, Metrics_time {metrics_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )

    return auc, logloss


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)


