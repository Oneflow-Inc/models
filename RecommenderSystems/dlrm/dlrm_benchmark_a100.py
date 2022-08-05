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
        default="int32",
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
    parser.add_argument("--eval_steps", type=int_list, default="1000000")
    parser.add_argument("--train_batch_size", type=int, default=55296)
    parser.add_argument("--learning_rate", type=float, default=24)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=15406)
    parser.add_argument("--decay_start", type=int, default=64163)
    parser.add_argument("--train_batches", type=int, default=75869)
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
    parser.add_argument("--store_type", type=str, default="device_mem")
    parser.add_argument("--cache_memory_budget_mb", type=int, default=8192)
    parser.add_argument("--amp", action="store_true", help="Run model with amp")
    parser.add_argument(
        "--split_allreduce", action="store_true", help="split bottom and top allreduce"
    )
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
                Dense(units[i], units[i + 1], not skip_final_activation or (i + 1) < num_layers)
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
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

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

        self.bottom_mlp = MLP(self.num_dense_fields, bottom_mlp, fused=use_fusedmlp)
        self.embedding = OneEmbedding(
            embedding_vec_size,
            persistent_path,
            table_size_array,
            one_embedding_store_type,
            cache_memory_budget_mb,
            one_embedding_key_type,
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
        )

    def forward(self, dense_fields, sparse_fields) -> flow.Tensor:
        if self.pad:
            dense_fields = flow.nn.functional.pad(dense_fields, self.pad, "constant")
        # dense_fields = flow.log(dense_fields + 1.0)
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
    )
    return model


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
    dense_loader = make_reader(f"{data_path}/dense_norm.bin", num_dense_fields, flow.float32)
    sparse_loader = make_reader(f"{data_path}/sparse.bin", num_sparse_fields, flow.int32)

    return label_loader, dense_loader, sparse_loader


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
    def __init__(self, dlrm_module, data_dir, batch_size, amp=False):
        super(DLRMValGraph, self).__init__()
        self.module = dlrm_module
        self.label_loader, self.dense_loader, self.sparse_loader = make_raw_dataloader(
            f"{data_dir}/val", batch_size, shuffle=False
        )

        if amp:
            self.config.enable_amp(True)

    def build(self):
        labels = self.label_loader()
        dense_fields = self.dense_loader()
        sparse_fields = self.sparse_loader()
        predicts = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        return labels, predicts.sigmoid()


class DLRMTrainGraph(flow.nn.Graph):
    def __init__(
        self,
        dlrm_module,
        loss,
        optimizer,
        data_dir,
        batch_size,
        lr_scheduler=None,
        grad_scaler=None,
        amp=False,
    ):
        super(DLRMTrainGraph, self).__init__()
        self.module = dlrm_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.label_loader, self.dense_loader, self.sparse_loader = make_raw_dataloader(
            f"{data_dir}/train", batch_size
        )
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self):
        labels = self.label_loader()
        dense_fields = self.dense_loader()
        sparse_fields = self.sparse_loader()
        logits = self.module(dense_fields.to("cuda"), sparse_fields.to("cuda"))
        loss = self.loss(logits, labels.to("cuda"))
        loss.backward()
        return loss.to("cpu")


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
    loss = flow.nn.BCEWithLogitsLoss(reduction="mean").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = DLRMValGraph(dlrm_module, args.data_dir, args.eval_batch_size, args.amp)
    train_graph = DLRMTrainGraph(
        dlrm_module,
        loss,
        opt,
        args.data_dir,
        args.train_batch_size,
        lr_scheduler,
        grad_scaler,
        args.amp,
    )

    dlrm_module.train()
    step, last_step, last_time = -1, 0, time.time()
    for step in range(1, args.train_batches + 1):
        loss = train_graph()
        if step % args.loss_print_interval == 0:
            loss = loss.numpy()
            if rank == 0:
                latency = (time.time() - last_time) / (step - last_step)
                throughput = args.train_batch_size / latency
                last_step, last_time = step, time.time()
                print(
                    f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, Latency "
                    + f"{(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {last_time}"
                )
            if np.isnan(loss):
                exit(1)

        if (args.eval_interval > 0 and step % args.eval_interval == 0) or (step in args.eval_steps):
            auc = eval(eval_graph, step)
            if args.save_model_after_each_eval:
                save_model(f"step_{step}_val_auc_{auc:0.5f}")
            dlrm_module.train()
            last_time = time.time()

    if args.eval_interval > 0 and step % args.eval_interval != 0:
        auc = eval(eval_graph, step)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")


def tensor_list_to_local(tensors):
    return (
        flow.cat(tensors, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )


def eval(eval_graph, cur_step=0):
    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()
    for i in range(args.eval_batches):
        label, pred = eval_graph()
        labels.append(label.to_local())
        preds.append(pred.to_local())

    labels = tensor_list_to_local(labels)
    preds = tensor_list_to_local(preds)

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
    #os.system("env")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    if args.split_allreduce:
        flow.boxing.nccl.set_fusion_max_ops_num(10)

    train(args)
