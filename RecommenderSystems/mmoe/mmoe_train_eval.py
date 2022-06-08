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


num_dense_fields = 11
num_sparse_fields = 29


class MmoeDataReader(object):
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

        column_names = [
            "age",
            "class_worker",
            "det_ind_code",
            "det_occ_code",
            "education",
            "wage_per_hour",
            "hs_college",
            "marital_stat",
            "major_ind_code",
            "major_occ_code",
            "race",
            "hisp_origin",
            "sex",
            "union_member",
            "unemp_reason",
            "full_or_part_emp",
            "capital_gains",
            "capital_losses",
            "stock_dividends",
            "tax_filer_stat",
            "region_prev_res",
            "state_prev_res",
            "det_hh_fam_stat",
            "det_hh_summ",
            "instance_weight",
            "mig_chg_msa",
            "mig_chg_reg",
            "mig_move_reg",
            "mig_same",
            "mig_prev_sunbelt",
            "num_emp",
            "fam_under_18",
            "country_father",
            "country_mother",
            "country_self",
            "citizenship",
            "own_or_self",
            "vet_question",
            "vet_benefits",
            "weeks_worked",
            "year",
            "income_50k",
        ]

        sparse_features = [
            "class_worker",
            "det_ind_code",
            "det_occ_code",
            "education",
            "hs_college",
            "major_ind_code",
            "major_occ_code",
            "race",
            "hisp_origin",
            "sex",
            "union_member",
            "unemp_reason",
            "full_or_part_emp",
            "tax_filer_stat",
            "region_prev_res",
            "state_prev_res",
            "det_hh_fam_stat",
            "det_hh_summ",
            "mig_chg_msa",
            "mig_chg_reg",
            "mig_move_reg",
            "mig_same",
            "mig_prev_sunbelt",
            "fam_under_18",
            "country_father",
            "country_mother",
            "country_self",
            "citizenship",
            "vet_question",
        ]

        dense_features = [
            col
            for col in column_names
            if col not in sparse_features and col not in ["income_50k", "marital_stat"]
        ]

        self.fields = dense_features + sparse_features + ["label_income", "label_marital"]
        self.num_fields = len(self.fields)
        self.dense_end = len(dense_features)
        self.sparse_end = len(dense_features + sparse_features)

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
                    dense = tail[0 : self.dense_end]
                    sparse = tail[self.dense_end : self.sparse_end]
                    label = tail[self.sparse_end :]
                    tail = None
                    yield np.stack(label, axis=-1), np.stack(dense, axis=-1), np.stack(
                        sparse, axis=-1
                    )
                else:
                    pos = 0
                    continue
            while (pos + batch_size) <= len(rglist[0]):
                dense = [rglist[j][pos : pos + batch_size] for j in range(0, self.dense_end)]
                sparse = [
                    rglist[j][pos : pos + batch_size]
                    for j in range(self.dense_end, self.sparse_end)
                ]
                label = [
                    rglist[j][pos : pos + batch_size]
                    for j in range(self.sparse_end, self.num_fields)
                ]
                pos += batch_size
                yield np.stack(label, axis=-1), np.stack(dense, axis=-1), np.stack(sparse, axis=-1)
            if pos != len(rglist[0]):
                tail = [rglist[i][pos:] for i in range(self.num_fields)]


def make_census_dataloader(data_path, batch_size, shuffle=True):
    """Make a Census-Income Parquet DataLoader.
    :return: a context manager when exit the returned context manager, the reader will be closed.
    """
    files = ["file://" + name for name in glob.glob(f"{data_path}/*.parquet")]
    files.sort()

    world_size = flow.env.get_world_size()
    batch_size_per_proc = batch_size // world_size

    return MmoeDataReader(
        files,
        batch_size_per_proc,
        None,  # TODO: iterate over all eval dataset
        shuffle_row_groups=shuffle,
        shard_seed=2019,
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
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
        )

    def forward(self, ids):
        return self.one_embedding.forward(ids)


class DNN(nn.Module):
    def __init__(
        self, in_features, hidden_units, out_features, skip_final_activation=False, dropout=0.0
    ) -> None:
        super(DNN, self).__init__()
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


class MmoeModule(nn.Module):
    def __init__(
        self,
        num_tasks=2,
        num_experts=3,
        embedding_vec_size=4,
        expert_dnn=[256, 128],
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        cache_memory_budget_mb=8192,
    ):
        super(MmoeModule, self).__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.embedding_layer = OneEmbedding(
            table_name="sparse_embedding",
            embedding_vec_size=embedding_vec_size,
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=3,
        )

        self.expert = DNN(
            in_features=embedding_vec_size * num_sparse_fields + num_dense_fields,
            hidden_units=expert_dnn[:-1],
            out_features=expert_dnn[-1],
            skip_final_activation=True,
            dropout=0.0,
        )

        self.experts = nn.ModuleList([])
        for _ in range(num_experts):
            expert_net = DNN(
                in_features=embedding_vec_size * num_sparse_fields + num_dense_fields,
                hidden_units=expert_dnn[:-1],
                out_features=expert_dnn[-1],
                skip_final_activation=True,
                dropout=0.0,
            )
            self.experts.append(expert_net)

        self.gates = nn.ModuleList([])
        self.towers = nn.ModuleList([])
        for _ in range(num_tasks):
            gate_net = nn.Linear(
                in_features=embedding_vec_size * num_sparse_fields + num_dense_fields,
                out_features=num_experts,
                bias=False,
            )
            self.gates.append(gate_net)

            tower_net = nn.Linear(in_features=expert_dnn[-1], out_features=1,)
            self.towers.append(tower_net)

    def forward(self, dense_inputs, sparse_inputs) -> flow.Tensor:
        sparse_emb = self.embedding_layer(sparse_inputs)
        inputs = flow.cat([sparse_emb.flatten(start_dim=1), dense_inputs], dim=1)
        # print("inputs: ", inputs.shape)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(inputs))
        expert_concat = flow.stack(expert_outs, dim=1)
        # print("expert_concat: ", expert_concat.shape)

        mmoe_outs = []
        for i in range(self.num_tasks):
            gate_out = self.gates[i](inputs).softmax()
            # print("gate: ", gate_out.shape)
            gate_out = gate_out.reshape([-1, self.num_experts, 1])
            # print("gate: ", gate_out.shape)
            gate_mul_expert = flow.mul(expert_concat, gate_out.expand_as(expert_concat))
            # print("gate_mul_expert: ", gate_mul_expert.shape)
            gate_mul_expert = flow.sum(gate_mul_expert, dim=1)
            # print("gate_mul_expert: ", gate_mul_expert.shape)

            tower_out = self.towers[i](gate_mul_expert)
            # print("tower: ", tower_out.shape)
            mmoe_outs.append(tower_out)

        return mmoe_outs


def make_mmoe_module(args):
    model = MmoeModule(
        num_tasks=2,
        num_experts=3,
        embedding_vec_size=4,
        expert_dnn=[256, 128],
        persistent_path=args.persistent_path,
        table_size_array=args.table_size_array,
        one_embedding_store_type=args.store_type,
        cache_memory_budget_mb=args.cache_memory_budget_mb,
    )
    return model


class MmoeTrainGraph(flow.nn.Graph):
    def __init__(
        self, mmoe_module, loss, optimizer, grad_scaler=None, amp=False, lr_scheduler=None,
    ):
        super(MmoeTrainGraph, self).__init__()
        self.module = mmoe_module
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
        label_income = labels[:, 0].unsqueeze(1)
        label_marital = labels[:, 1].unsqueeze(1)
        loss_income = self.loss(logits[0], label_income.to("cuda"))
        loss_marital = self.loss(logits[1], label_marital.to("cuda"))
        loss = loss_income + loss_marital
        loss.backward()
        return loss.to("cpu")


class MmoeValGraph(flow.nn.Graph):
    def __init__(self, mmoe_module, amp=False):
        super(MmoeValGraph, self).__init__()
        self.module = mmoe_module
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        predicts = self.module(features.to("cuda"))
        return predicts.sigmoid()


def make_lr_scheduler(args, optimizer):
    batches_per_epoch = math.ceil(args.num_train_samples / args.batch_size)
    milestones = [
        batches_per_epoch * (i + 1)
        for i in range(math.floor(math.log(args.min_lr / args.learning_rate, args.lr_factor)))
    ]
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=args.lr_factor,
    )

    return multistep_lr


def train(args):
    rank = flow.env.get_rank()

    mmoe_module = make_mmoe_module(args)
    mmoe_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    def load_model(dir):
        if rank == 0:
            print(f"Loading model from {dir}")
        if os.path.exists(dir):
            state_dict = flow.load(dir, global_src_rank=0)
            mmoe_module.load_state_dict(state_dict, strict=False)
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
        state_dict = mmoe_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    # TODO: clip gradient norm
    opt = flow.optim.Adam(mmoe_module.parameters(), lr=args.learning_rate)
    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="mean").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )

    eval_graph = MmoeValGraph(mmoe_module, args.amp)
    train_graph = MmoeTrainGraph(
        mmoe_module, loss, opt, grad_scaler, args.amp, lr_scheduler=lr_scheduler
    )

    batches_per_epoch = math.ceil(args.num_train_samples / args.batch_size)

    cached_eval_batches = prefetch_eval_batches(
        f"{args.data_dir}/test", args.batch_size, math.ceil(args.num_test_samples / args.batch_size)
    )

    mmoe_module.train()
    epoch = 0
    with make_census_dataloader(f"{args.data_dir}/train", args.batch_size) as loader:
        step, last_step, last_time = -1, 0, time.time()
        for step in range(1, args.train_batches + 1):
            labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            # print("label: ", labels.shape)
            # print("dense: ", dense_fields.shape)
            # print("sparse: ", sparse_fields.shape)
            loss = train_graph(labels, dense_fields, sparse_fields)
            if step % args.loss_print_interval == 0:
                loss = loss.numpy()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}], Step {step}, Loss {loss:0.4f}, "
                        + f"Latency {(latency * 1000):0.3f} ms, Throughput {throughput:0.1f}, {strtime}"
                    )

            # if step % batches_per_epoch == 0:
            #     epoch += 1
            #     auc = eval(
            #         args,
            #         eval_graph,
            #         cur_step=step,
            #         epoch=epoch,
            #         cached_eval_batches=cached_eval_batches,
            #     )
            #     if args.save_model_after_each_eval:
            #         save_model(f"step_{step}_val_auc_{auc:0.5f}")

            # mmoe_module.train()
            # last_time = time.time()

    # if step % batches_per_epoch != 0:
    #     auc = eval(
    #         args,
    #         eval_graph,
    #         cur_step=step,
    #         epoch=epoch,
    #         cached_eval_batches=cached_eval_batches,
    #     )
    #     if args.save_model_after_each_eval:
    #         save_model(f"step_{step}_val_auc_{auc:0.5f}")


def np_to_global(np):
    t = flow.from_numpy(np)
    return t.to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))


def batch_to_global(np_label, np_dense, np_sparse, is_train=True):
    labels = np_to_global(np_label) if is_train else np_label
    np_dense = np_to_global(np_dense)
    np_sparse = np_to_global(np_sparse)

    return labels, np_dense, np_sparse


def prefetch_eval_batches(data_dir, batch_size, num_batches):
    cached_eval_batches = []
    with make_census_dataloader(data_dir, batch_size, shuffle=False) as loader:
        for _ in range(num_batches):
            labels, dense_fields, sparse_fields = batch_to_global(*next(loader))
            cached_eval_batches.append((labels, dense_fields, sparse_fields))
    return cached_eval_batches


def eval(args, eval_graph, cur_step=0, epoch=0, cached_eval_batches=None):
    batches_per_epoch = math.ceil(args.num_test_samples / args.batch_size)

    eval_graph.module.eval()
    labels, preds = [], []
    eval_start_time = time.time()

    for i in range(batches_per_epoch):
        label, features = cached_eval_batches[i]
        pred = eval_graph(features)
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

    metrics_start_time = time.time()
    auc = flow.roc_auc_score(labels, preds).numpy()[0]
    logloss = flow._C.binary_cross_entropy_loss(preds, labels, weight=None, reduction="mean")
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

    return auc


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    args = get_args()
    train(args)
