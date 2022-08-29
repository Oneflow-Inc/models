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
import random
import pickle
from pkl_reader import DataInput, DataInputTest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--num_train_samples", type=int, default=2608764, help="the number of train samples",
    )
    parser.add_argument(
        "--num_test_samples", type=int, default=384806, help="the number of validation samples",
    )

    parser.add_argument("--model_load_dir", type=str, default=None, help="model loading directory")
    parser.add_argument("--model_save_dir", type=str, default=None, help="model saving directory")
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not",
    )
    parser.add_argument(
        "--save_model_after_each_eval",
        action="store_true",
        help="save model after each eval or not",
    )

    parser.add_argument("--max_len", type=int, default=512, help="max sequence length")
    parser.add_argument("--embedding_size", type=int, default=64, help="embedding vector size")
    parser.add_argument(
        "--attention_layer_hidden_dim",
        type=int_list,
        default="80,40",
        help="attention layer hidden units number",
    )

    parser.add_argument("--lr_factor", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--learning_rate", type=float, default=1, help="learning rate")
    parser.add_argument("--optim", type=str, default="SGD", help="optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="training/evaluation batch size")
    parser.add_argument(
        "--train_batches", type=int, default=652192, help="the maximum number of training batches",
    )
    parser.add_argument("--loss_print_interval", type=int, default=1000, help="")

    parser.add_argument(
        "--table_size_array",
        type=int_list,
        default="63001,801",
        help="embedding table size array for sparse fields",
    )
    parser.add_argument(
        "--persistent_path", type=str, default="./persistent", help="path for persistent kv store",
    )
    parser.add_argument(
        "--store_type",
        type=str,
        default="device_mem",
        help="OneEmbeddig persistent kv store type: device_mem, cached_host_mem, cached_ssd",
    )
    parser.add_argument(
        "--cache_memory_budget_mb",
        type=int,
        default=1024,
        help="size of cache memory budget on each device in megabytes when store_type is cached_host_mem or cached_ssd",
    )

    parser.add_argument(
        "--amp", action="store_true", help="enable Automatic Mixed Precision(AMP) training or not",
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
        padding_idx, 
    ):
        assert table_size_array is not None
        vocab_size = sum(table_size_array)

        # todo: last column should be initialized to 0
        b = [math.sqrt(6 / (table_size + embedding_vec_size[0])) for table_size in table_size_array]
        embd_initializer = [flow.one_embedding.make_uniform_initializer(-a, a) for a in b]
        embd_columns = [flow.one_embedding.make_column_options(e) for e in embd_initializer]

        bias_initializer = flow.one_embedding.make_constant_initializer(0)
        bias_column = flow.one_embedding.make_column_options(bias_initializer)
        tables = [
            flow.one_embedding.make_table_options([embd_column, bias_column])
            for embd_column in embd_columns
        ]

        if store_type == "device_mem":
            store_options = flow.one_embedding.make_device_mem_store_options(
                persistent_path=persistent_path, capacity=vocab_size, size_factor=size_factor, padding_idx=padding_idx
            )
        elif store_type == "cached_host_mem":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_host_mem_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
                padding_idx=padding_idx, 
            )
        elif store_type == "cached_ssd":
            assert cache_memory_budget_mb > 0
            store_options = flow.one_embedding.make_cached_ssd_store_options(
                cache_budget_mb=cache_memory_budget_mb,
                persistent_path=persistent_path,
                capacity=vocab_size,
                size_factor=size_factor,
                padding_idx=padding_idx, 
            )
        else:
            raise NotImplementedError("not support", store_type)

        super(OneEmbedding, self).__init__()
        self.one_embedding = flow.one_embedding.MultiTableMultiColumnEmbedding(
            name=table_name,
            embedding_dim=embedding_vec_size,
            dtype=flow.float,
            key_type=flow.int64,
            tables=tables,
            store_options=store_options,
            padding_idx=padding_idx, 
        )

    def forward(self, ids, table_ids):
        return self.one_embedding.forward(ids, table_ids=table_ids)


class DNN(nn.Module):
    def __init__(
        self, in_features, hidden_units, out_features, skip_final_activation=False, dropout=0.0,
    ) -> None:
        super(DNN, self).__init__()
        denses = []
        dropout_rates = [dropout] * len(hidden_units) + [0.0]
        use_relu = [True] * len(hidden_units) + [not skip_final_activation]
        hidden_units = [in_features] + hidden_units + [out_features]
        for idx in range(len(hidden_units) - 1):
            denses.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=True))
            if use_relu[idx]:
                denses.append(nn.Sigmoid())
            if dropout_rates[idx] > 0:
                denses.append(nn.Dropout(p=dropout_rates[idx]))
        self.linear_layers = nn.Sequential(*denses)

        for name, param in self.linear_layers.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                param.data.fill_(0.0)

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        return self.linear_layers(x)


class DINModule(nn.Module):
    def __init__(
        self,
        embedding_size=64,
        attention_layer_hidden_dim=[80, 40],
        second_con_layer_hidden_dim=[80, 40],
        persistent_path=None,
        table_size_array=None,
        one_embedding_store_type="cached_host_mem",
        cache_memory_budget_mb=8192,
        size_factor=1,
        max_len=32,
        cate_list=None,
        padding_idx=0, 
    ):
        super(DINModule, self).__init__()

        self.cate_list = cate_list
        self.arange = flow.arange(max_len).to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

        self.embedding_size = embedding_size
        self.num_items = table_size_array[0]

        self.firInDim = 2 * self.embedding_size
        self.firOutDim = 2 * self.embedding_size
        self.embedding = OneEmbedding(
            table_name="oneembedding",
            embedding_vec_size=[embedding_size, 1],
            persistent_path=persistent_path,
            table_size_array=table_size_array,
            store_type=one_embedding_store_type,
            cache_memory_budget_mb=cache_memory_budget_mb,
            size_factor=size_factor,
            padding_idx=padding_idx, 
        )
        self.table_ids = flow.tensor(
            [0] + [1] + [0] * max_len + [1] * max_len, dtype=flow.int64
        ).to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

        self.attention_layer_input_dim = self.embedding_size * 8
        self.attention_layer = DNN(
            in_features=self.attention_layer_input_dim,
            hidden_units=attention_layer_hidden_dim,
            out_features=1,
            skip_final_activation=True,
        )
        self.attention_softmax = flow.nn.Softmax(dim = 2)

        self.bn1 = flow.nn.BatchNorm1d(2 * self.embedding_size)
        self.first_con_layer = DNN(
            in_features=self.firInDim,
            hidden_units=[],
            out_features=self.firOutDim,
            skip_final_activation=True,
        )
        
        self.bn2 = flow.nn.BatchNorm1d(6 * self.embedding_size)
        self.second_con_layer_input_dim = 6 * self.embedding_size
        self.second_con_layer = DNN(
            in_features=self.second_con_layer_input_dim,
            hidden_units=second_con_layer_hidden_dim,
            out_features=1,
            skip_final_activation=True,
        )

    def forward(self, inputs) -> flow.Tensor:
        target_item, hist_item_seq, seq_len = inputs
        # (b, 1) (b, s) (b, 1)
        b, s = hist_item_seq.shape
        e = self.embedding_size

        target_cat = self.cate_list[target_item] + 63001# (b, 1)
        hist_cat_seq = self.cate_list[hist_item_seq] + 63001# (b, s)
        #mask = -1e9 * (self.arange[None, :] >= seq_len[:, None]) # (b, s)

        ids = flow.cat(
            [target_item, target_cat, hist_item_seq, hist_cat_seq], dim=1
        )  # (b, 1+1+s+s)
        table_ids = self.table_ids.unsqueeze(0).expand(b, -1)
        embeddings = self.embedding(ids, table_ids)  # (b, 1+1+s+s, e+1)

        target_item_emb = embeddings[:, 0, :e]
        target_cat_emb = embeddings[:, 1, :e]
        hist_item_emb = embeddings[:, 2 : (2 + s), :e]
        hist_cat_emb = embeddings[:, (2 + s) : (2 + s + s), :e]
        item_b = embeddings[:, 0, -1]
        target_item_seq_emb = target_item_emb.unsqueeze(1).expand(-1, s, -1)
        target_cat_seq_emb = target_cat_emb.unsqueeze(1).expand(-1, s, -1)

        hist_seq_concat = flow.concat([hist_item_emb, hist_cat_emb], dim=2)
        target_seq_concat = flow.concat([target_item_seq_emb, target_cat_seq_emb], dim=2)
        target_concat = flow.concat([target_item_emb, target_cat_emb], dim=1)
        concat = flow.concat(
            [
                target_seq_concat,
                hist_seq_concat,
                target_seq_concat - hist_seq_concat,
                hist_seq_concat * target_seq_concat,
            ],
            dim=2,
        )

        concat = self.attention_layer(concat) #(b, s, 1)
        key_masks = self.arange[None, :] < seq_len[:, None] # (b, s)
        key_masks = key_masks.unsqueeze(-1) # (b, s, 1)
        paddings = flow.ones_like(concat) * (-2 ** 32 + 1)

        atten_fc3 = flow.where(key_masks, concat, paddings) # paddle use + but not where
        atten_fc3 = flow.transpose(atten_fc3, perm=[0, 2, 1]) #(b, 1, s)
        #atten_fc3 /= self.firInDim ** -0.5 #(b, 1, s) #error
        atten_fc3 /= self.firInDim ** 0.5 #(b, 1, s) #error

        #weight = flow.nn.functional.softmax(atten_fc3) #(b, 1, s) #error
        weight = self.attention_softmax(atten_fc3) #(b, 1, s)
        output = flow.matmul(weight, hist_seq_concat) #(b, 1, e)
        output = flow.squeeze(output) #(b, e)
        #output = self.bn1(output)

        concat = self.first_con_layer(output)
        embedding_concat = flow.concat([concat, target_concat, concat * target_concat], dim=1)
        #embedding_concat = self.bn2(embedding_concat)
        embedding_concat = self.second_con_layer(embedding_concat)
        logit = embedding_concat + item_b.unsqueeze(1)
        # return logit.sigmoid()
        return logit 


def make_din_module(args, cate_list, padding_idx):
    model = DINModule(
        embedding_size=args.embedding_size,
        attention_layer_hidden_dim=args.attention_layer_hidden_dim,
        persistent_path=args.persistent_path,
        table_size_array=args.table_size_array,
        one_embedding_store_type=args.store_type,
        cache_memory_budget_mb=args.cache_memory_budget_mb,
        size_factor=1 if args.optim == "SGD" else 3,
        max_len=args.max_len,
        cate_list=cate_list,
        padding_idx=padding_idx
    )
    return model


class DINValGraph(flow.nn.Graph):
    def __init__(self, din_module, amp=False):
        super(DINValGraph, self).__init__()
        self.module = din_module
        if amp:
            self.config.enable_amp(True)

    def build(self, features):
        for i in range(len(features)):
            features[i] = features[i].to("cuda")
        predicts = self.module(features)
        p_n = predicts.view(-1, 2)
        x = p_n[:, 0] - p_n[:, 1]
        mf_auc = x.sum()
        return predicts.sigmoid(), mf_auc
        # return predicts


class DINTrainGraph(flow.nn.Graph):
    def __init__(
        self, din_module, loss, optimizer, grad_scaler=None, amp=False, lr_scheduler=None,
    ):
        super(DINTrainGraph, self).__init__()
        self.module = din_module
        self.loss = loss
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_model_update_ops(True)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_cast_scale(True)
        if amp:
            self.config.enable_amp(True)
            self.set_grad_scaler(grad_scaler)

    def build(self, labels, features):
        for i in range(len(features)):
            features[i] = features[i].to("cuda")
        logits = self.module(features)
        loss = self.loss(logits, labels.to(dtype=flow.float32, device="cuda"))
        #loss = flow.mean(loss)
        loss.backward()
        return loss.to("cpu") 


def make_lr_scheduler(args, optimizer):
    batches_per_epoch = math.ceil(args.num_train_samples / args.batch_size)
    milestones = [
        batches_per_epoch * (i + 1)
        for i in range(math.floor(math.log(args.min_lr / args.learning_rate, args.lr_factor)))
    ] #[81524, 163048, 244572, 326096, 407620, 489144]
    milestones = [336000]
    multistep_lr = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=args.lr_factor,
    )

    return multistep_lr


def prefetch_eval_batches(dataset, batch_size):
    cached_batches = []
    for _, uij in DataInputTest(dataset, batch_size, max_sl=512):
        label, features = batch_to_global(*uij)
        cached_batches.append((label, features))
    return cached_batches


def train(args):
    rank = flow.env.get_rank()
    padding_idx = 0

    print("load dataset form", f'{args.data_dir}/dataset.pkl')
    with open(f'{args.data_dir}/dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)
        #cate_list += item_count 
        cate_list = flow.tensor(cate_list, dtype=flow.int64).to_global(
            placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.broadcast,
        )
        print(cate_list)

    din_module = make_din_module(args, cate_list, padding_idx)
    din_module.to_global(flow.env.all_device_placement("cuda"), flow.sbp.broadcast)

    def load_model(dir):
        if rank == 0:
            print(f"Loading model from {dir}")
        if os.path.exists(dir):
            state_dict = flow.load(dir, global_src_rank=0)
            din_module.load_state_dict(state_dict, strict=True)
            # din_module.load_state_dict(state_dict, strict=True)
            print("din_module.load_state_dict(state_dict, strict=True)")
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
        state_dict = din_module.state_dict()
        flow.save(state_dict, save_path, global_dst_rank=0)

    if args.save_initial_model:
        save_model("initial_checkpoint")

    # TODO: clip gradient norm
    if args.optim == "SGD":
        opt = flow.optim.SGD(din_module.parameters(), lr=args.learning_rate)
    elif args.optim == "Adam":
        opt = flow.optim.Adam(din_module.parameters(), lr=args.learning_rate)
    else:
        print("Only support SGD or Adam")
        exit()

    lr_scheduler = make_lr_scheduler(args, opt)
    loss = flow.nn.BCEWithLogitsLoss(reduction="mean").to("cuda")
    # loss = flow.nn.BCELoss(reduction="mean").to("cuda")

    if args.loss_scale_policy == "static":
        grad_scaler = flow.amp.StaticGradScaler(1024)
    else:
        grad_scaler = flow.amp.GradScaler(
            init_scale=1073741824, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
        )


    eval_graph = DINValGraph(din_module, args.amp)
    train_graph = DINTrainGraph(
        din_module, loss, opt, grad_scaler, args.amp, lr_scheduler=lr_scheduler 
    )

    din_module.train()
    epoch = 0

    def pprint(n, v):
        a = v.numpy()
        print(n, a.mean(), a.std(), a.max(), a.min(), v.shape)
        
    for n, v in din_module.named_parameters():
        pprint(n, v)
    for n, v in din_module.named_buffers():
        pprint(n, v)

    cached_eval_batches = prefetch_eval_batches(test_set, args.batch_size)
    auc = eval(cached_eval_batches, eval_graph, cur_step=0, epoch=0)
    #exit()
    if args.save_model_after_each_eval:
        save_model(f"step_0_val_auc_{auc:0.5f}")
    step, last_step, last_time, losses = 0, 0, time.time(), []
    for epoch in range(10):
        random.shuffle(train_set)
        for _, uij in DataInput(train_set, args.batch_size, max_sl=args.max_len):
            step += 1
            labels, features = batch_to_global(*uij)
            loss = train_graph(labels, features)
            losses.append(loss.view([1]))
            #print(loss, uij[0], w.numpy(), w.shape)
            #np.save("of_watch.npy", w.numpy())
            #print(loss.numpy())
            #if step >= 1000:
            #    exit()
            if step % args.loss_print_interval == 0:
                losses = tensor_list_to_local(losses)
                loss = losses.mean()
                if rank == 0:
                    latency = (time.time() - last_time) / (step - last_step)
                    throughput = args.batch_size / latency
                    last_step, last_time = step, time.time()
                    strtime = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"Rank[{rank}] Epoch {epoch} Step {step} Loss {loss} "
                        + f"Latency {(latency * 1000):0.3f} ms Throughput {throughput:0.1f} {strtime}"
                    )
                losses = []
        auc = eval(cached_eval_batches, eval_graph, cur_step=step, epoch=epoch)
        if args.save_model_after_each_eval:
            save_model(f"step_{step}_val_auc_{auc:0.5f}")
        din_module.train()
        last_time = time.time()


def batch_to_global(u, i, y, hist_i, sl):
    # u is ignored
    label = flow.Tensor(y).to_global(
        placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
    ) 
    target_item = flow.tensor(i, dtype=flow.int64).to_global(
        placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
    ) 
    hist_item_seq = flow.tensor(hist_i, dtype=flow.int64).to_global(
        placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
    ) 
    seq_len = flow.tensor(sl, dtype=flow.int32).to_global(
        placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0)
    ) 
    return label.view(-1, 1), [target_item.view(-1, 1), hist_item_seq, seq_len] 


def tensor_list_to_local(tensors):
    return (
        flow.cat(tensors, dim=0)
        .to_global(placement=flow.env.all_device_placement("cpu"), sbp=flow.sbp.split(0))
        .to_global(sbp=flow.sbp.broadcast())
        .to_local()
    )


def calc_auc(labels, preds):
    score_arr = flow.cat([1-labels, labels, preds], dim=1)
    print(score_arr.shape)
    score_arr = score_arr.numpy()
    arr = sorted(score_arr, key=lambda d:d[2])
    auc = 0.0
    fp1, tp1, fp2, tp2 = 0.0, 0.0, 0.0, 0.0
    for record in arr:
        fp2 += record[0] # noclick
        tp2 += record[1] # click
        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1, tp1 = fp2, tp2

    # if all nonclick or click, disgard
    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def eval(cached_batches, eval_graph, cur_step=0, epoch=0):
    eval_graph.module.eval()
    labels, preds, mf_aucs = [], [], []
    eval_start_time = time.time()

    for label, features in cached_batches:
        pred, mf_auc = eval_graph(features)
        labels.append(label)
        preds.append(pred.to_local())
        mf_aucs.append(mf_auc.view([1]).to_local())

        #flow.from_numpy(np.concatenate(labels, axis=0))
    labels = tensor_list_to_local(labels)
    preds = tensor_list_to_local(preds)
    mf_aucs = tensor_list_to_local(mf_aucs)

    flow.comm.barrier()
    gauc = mf_aucs.sum() / labels.shape[0]
    eval_time = time.time() - eval_start_time

    rank = flow.env.get_rank()

    metrics_start_time = time.time()
    auc = flow.roc_auc_score(labels, preds).numpy()[0]
    auc1 = calc_auc(labels, preds)
    metrics_time = time.time() - metrics_start_time
    if rank == 0:
        host_mem_mb = psutil.Process().memory_info().rss // (1024 * 1024)
        stream = os.popen("nvidia-smi --query-gpu=memory.used --format=csv")
        device_mem_str = stream.read().split("\n")[rank + 1]

        strtime = time.strftime("%Y-%m-%d %H:%M:%S")
        print(
                f"Rank[{rank}], Epoch {epoch}, Step {cur_step}, AUC {auc:0.6f}, AUC {auc1:0.6f}, "
            + f"Eval_time {eval_time:0.2f} s, Metrics_time {metrics_time:0.2f} s, Eval_samples {labels.shape[0]}, "
            + f"GPU_Memory {device_mem_str}, Host_Memory {host_mem_mb} MiB, {strtime}"
        )
    return auc


if __name__ == "__main__":
    os.system(sys.executable + " -m oneflow --doctor")
    flow.boxing.nccl.enable_all_to_all(True)
    flow.set_printoptions(precision=8)
    args = get_args()
    train(args)
