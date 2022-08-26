"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import os
import sys
import glob
import time
import numpy as np
import psutil
import warnings
import oneflow as flow
import oneflow.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)
from petastorm.reader import make_batch_reader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--disable_fusedmlp", action="store_true", help="disable fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=128)
    parser.add_argument(
        "--one_embedding_key_type",
        type=str,
        default="int64",
        help="OneEmbedding key type: int32, int64",
    )
    parser.add_argument("--bottom_mlp", type=int_list, default="512,256,128")
    parser.add_argument("--top_mlp", type=int_list, default="1024,1024,512,256")
    parser.add_argument(
        "--disable_interaction_padding",
        action="store_true",
        help="disable interaction padding or not",
    )
    parser.add_argument(
        "--disable_dense_input_padding",
        action="store_true",
        help="disable dense input padding or not",
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
    parser.add_argument("--eval_batches", type=int, default=1612, help="number of eval batches")
    parser.add_argument("--eval_batch_size", type=int, default=55296)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=55296)
    parser.add_argument("--learning_rate", type=float, default=24)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=27772)
    parser.add_argument("--decay_start", type=int, default=49315)
    parser.add_argument("--train_batches", type=int, default=75000)
    parser.add_argument("--loss_print_interval", type=int, default=1000)
    parser.add_argument(
        "--table_size_array",
        type=int_list,
        default="39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36",
        help="Embedding table size array for sparse fields",
    )
    parser.add_argument(
        "--persistent_path", type=str, required=True, help="path for persistent kv store",
    )
    parser.add_argument("--store_type", type=str, default="cached_host_mem")
    parser.add_argument("--cache_memory_budget_mb", type=int, default=8192)
    parser.add_argument("--amp", action="store_true", help="Run model with amp")
    parser.add_argument("--loss_scale_policy", type=str, default="static", help="static or dynamic")
    parser.add_argument("--freeze_embedding", action="store_true", help="freeze embedding model")
    parser.add_argument(
        "--freeze_backbone", action="store_true", help="freeze backbone models except embedding"
    )

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
                        np.concatenate((tail[i], rglist[i][0 : (batch_size - len(tail[i]))]))
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
                dense = [rglist[j][pos : pos + batch_size] for j in range(1, self.I_end)]
                sparse = [rglist[j][pos : pos + batch_size] for j in range(self.I_end, self.C_end)]
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
        self,
        in_features: int,
        hidden_units,
        skip_final_activation=False,
        fused=True,
        freeze_model=False,
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
                Dense(units[i], units[i + 1], not skip_final_activation or (i + 1) < num_layers,)
                for i in range(num_layers)
            ]
            self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, 0.0, np.sqrt(2 / sum(param.shape)))
            elif "bias" in name:
                nn.init.normal_(param, 0.0, np.sqrt(1 / param.shape[0]))
            if freeze_model:
                print(f"model {name} is freezed")
                param.requires_grad = False

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
        n_cols = num_embedding_fields + 2 if self.interaction_itself else num_embedding_fields + 1
        output_size = dense_feature_size + sum(range(n_cols))
        self.output_size = ((output_size + 8 - 1) // 8 * 8) if interaction_padding else output_size
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
    def __init__(
        self,
        embedding_vec_size,
        persistent_path,
        table_size_array,
        store_type,
        cache_memory_budget_mb,
        key_type,
        freeze_model,
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)
        assert key_type in ["int32", "int64"], "OneEmbedding key_type must be integers"

        scales = np.sqrt(1 / np.array(table_size_array))
        tables = [
            flow.one_embedding.make_table_options(
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
            key_type=getattr(flow, key_type),
            tables=tables,
            store_options=store_options,
        )
        for name, param in self.one_embedding.named_parameters():
            if freeze_model:
                print(f"model {name} is freezed")
                param.requires_grad = False

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class DLRMModule(nn.Module):
    def __init__(
        self,
        embedding_vec_size=128,
        bottom_mlp=[512, 256, 128],
        top_mlp=[1024, 1024, 512, 256],
        use_fusedmlp=True,
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        one_embedding_key_type="int64",
        cache_memory_budget_mb=8192,
        interaction_itself=True,
        interaction_padding=True,
        dense_input_padding=True,
        freeze_embedding=False,
        freeze_backbone=False,
    ):
        super(DLRMModule, self).__init__()
        assert (
            embedding_vec_size == bottom_mlp[-1]
        ), "Embedding vector size must equle to bottom MLP output size"
        self.num_dense_fields = (
            ((num_dense_fields + 8 - 1) // 8 * 8) if dense_input_padding else num_dense_fields
        )
        self.pad = (
            [0, self.num_dense_fields - num_dense_fields]
            if self.num_dense_fields > num_dense_fields
            else None
        )

        self.bottom_mlp = MLP(
            self.num_dense_fields, bottom_mlp, fused=use_fusedmlp, freeze_model=freeze_backbone
        )
        self.embedding = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            table_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb,
            one_embedding_key_type,
            freeze_embedding,
        )
        self.interaction = Interaction(
            bottom_mlp[-1],
            num_sparse_fields,
            interaction_itself,
            interaction_padding=interaction_padding,
        )
        self.top_mlp = MLP(
            self.interaction.output_size,
            top_mlp + [1],
            skip_final_activation=True,
            fused=use_fusedmlp,
            freeze_model=freeze_backbone,
        )

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        if self.pad:
            dense_fields = flow.nn.functional.pad(dense_fields, self.pad, "constant")
        dense_fields = flow.log(dense_fields + 1.0)
        dense_fields = self.bottom_mlp(dense_fields)
        embedding = self.embedding(sparse_fields)
        features = self.interaction(dense_fields, embedding)
        return self.top_mlp(features)


def make_dlrm_module(args):
    model = DLRMModule(
        embedding_vec_size=args.embedding_vec_size,
        bottom_mlp=args.bottom_mlp,
        top_mlp=args.top_mlp,
        use_fusedmlp=not args.disable_fusedmlp,
        persistent_path=args.persistent_path,
        table_size_array=args.table_size_array,
        one_embedding_store_type=args.store_type,
        one_embedding_key_type=args.one_embedding_key_type,
        cache_memory_budget_mb=args.cache_memory_budget_mb,
        interaction_itself=args.interaction_itself,
        interaction_padding=not args.disable_interaction_padding,
        dense_input_padding=not args.disable_dense_input_padding,
        freeze_embedding=args.freeze_embedding,
        freeze_backbone=args.freeze_backbone,
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
        optimizer, decay_batch=args.decay_batches, end_learning_rate=0, power=2.0, cycle=False,
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
        self, dlrm_module, loss, optimizer, lr_scheduler=None, grad_scaler=None, amp=False,
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
            label, dense_fields, sparse_fields = batch_to_global(*next(loader), is_train=False)
            cached_eval_batches.append((label, dense_fields, sparse_fields))
    return cached_eval_batches


def train(args):
    rank = flow.env.get_rank()

    dlrm_module = make_dlrm_module(args)
    dlrm_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

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

    opt = flow.optim.SGD(dlrm_module.parameters(), lr=args.learning_rate)
    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="none").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DLRMValGraph(dlrm_module, args.amp)
    train_graph = DLRMTrainGraph(dlrm_module, loss, opt, lr_scheduler, grad_scaler, args.amp)

    cached_eval_batches = prefetch_eval_batches(
        f"{args.data_dir}/test", args.eval_batch_size, args.eval_batches
    )

    dlrm_module.train()
    with make_criteo_dataloader(f"{args.data_dir}/train", args.train_batch_size) as loader:
        step, last_step, last_time = -1, 0, time.time()
        for step in range(1, args.train_batches + 1):
            labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            loss = train_graph(labels, dense_fields, sparse_fields)
            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.train_batch_size / latency
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
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")


def np_to_global(np):
    t = flow.from_numpy(np)
    return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))


def batch_to_global(np_label, np_dense, np_sparse, is_train=True):
    labels = np_to_global(np_label.reshape(-1, 1)) if is_train else np_label.reshape(-1, 1)
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
        np_to_global(np.concatenate(labels, axis=0)).to_global(sbp=flow.sbp.broadcast()).to_local()
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
    args = get_args()

    train(args)
