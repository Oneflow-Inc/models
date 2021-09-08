import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import oneflow as flow

from config import get_args
from dataloader_utils_consistent import OFRecordDataLoader
from wide_and_deep_module_dmp import WideAndDeep
from util import dump_to_npy, save_param_npy

world_size = flow.env.get_world_size()
placement = flow.placement("cpu", {0: range(world_size)})

def prepare_modules(args):
    
    sbp = flow.sbp.broadcast
    train_dataloader = OFRecordDataLoader(
        args, data_root=args.data_dir, batch_size=args.batch_size,placement=placement,sbp=sbp
    )
    val_dataloader = OFRecordDataLoader(args, data_root=args.data_dir, mode="val",batch_size=args.batch_size,placement=placement,sbp=sbp)

    wdl_module = WideAndDeep(args)
    wdl_module = wdl_module.to("cuda")


    if args.model_load_dir != "":
        print("load checkpointed model from ", args.model_load_dir)
        wdl_module.load_state_dict(flow.load(args.model_load_dir))

    if args.save_initial_model and args.model_save_dir != "":
        path = os.path.join(args.model_save_dir, "initial_checkpoint")
        if not os.path.isdir(path):
            flow.save(wdl_module.state_dict(), path)

    bce_loss = flow.nn.BCELoss(reduction="mean")
    bce_loss.to("cuda")

    opt = flow.optim.SGD(wdl_module.parameters(), lr=args.learning_rate, momentum=0.9)
    return train_dataloader, val_dataloader, wdl_module, bce_loss, opt


def print_eval_metrics(step, loss, lables_list, predicts_list):
    all_labels = np.concatenate(lables_list, axis=0)
    all_predicts = np.concatenate(predicts_list, axis=0)
    auc = (
        "NaN"
        if np.isnan(all_predicts).any()
        else roc_auc_score(all_labels, all_predicts)
    )
    rank=flow.env.get_rank()
    print(f"device {rank}: iter {step} eval_loss {loss} auc {auc}")


world_size = flow.env.get_world_size()
placement = flow.placement("cpu", {0: range(world_size)})

if __name__ == "__main__":
    args = get_args()

    train_dataloader, val_dataloader, wdl_module, bce_loss, opt = prepare_modules(args)

    class WideAndDeepGraph(flow.nn.Graph):
        def __init__(self, dataloader):
            super(WideAndDeepGraph, self).__init__()
            self.module = wdl_module
            self.dataloader = dataloader
            self.bce_loss = bce_loss

        def build(self):
            with flow.no_grad():
                return self.graph()

        def graph(self):
            (
                labels,
                dense_fields,
                wide_sparse_fields,
                deep_sparse_fields,
            ) = self.dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = self.module(dense_fields, wide_sparse_fields, deep_sparse_fields)
            loss = self.bce_loss(predicts, labels)
            #predicts是broadcast,labels是broadcast,loss是broadcast(若predicts是split(0)，labels是broadcast，那么to_local后数据len不一样)
            # print('predicts',predicts)
            # print('labels',labels)
            # print('loss',loss)


            return predicts, labels, loss

    class WideAndDeepTrainGraph(WideAndDeepGraph):
        def __init__(self, dataloader):
            super(WideAndDeepTrainGraph, self).__init__(dataloader)
            self.add_optimizer(opt)

        def build(self):
            predicts, labels, loss = self.graph()
            loss.backward()
            return predicts, labels, loss

    eval_graph = WideAndDeepGraph(val_dataloader)
    train_graph = WideAndDeepTrainGraph(train_dataloader)

    losses = []
    wdl_module.train()

    for i in tqdm(range(args.max_iter)):
        predicts, labels, train_loss = train_graph()
        #train_loss是broadcast
        losses.append(train_loss.to_local().numpy().mean())

        if (i + 1) % args.print_interval == 0:
            l = sum(losses) / len(losses)
            losses = []
            rank=flow.env.get_rank()
            print(f"device {rank}: iter {i+1} train_loss {l} time {time.time()}")
            if args.eval_batchs <= 0:
                continue

            eval_loss_acc = 0.0
            lables_list = []
            predicts_list = []
            wdl_module.eval()
            for j in range(args.eval_batchs):
                predicts, labels, eval_loss = eval_graph()
                losses.append(eval_loss.to_local().numpy().mean())
                eval_loss_acc += eval_loss.to_local().numpy().mean()
                #print('labels',labels)
                #print('predicts',predicts)
                lables_list.append(labels.to_local().numpy())
                predicts_list.append(predicts.to_local().numpy())

            print_eval_metrics(
                i + 1, eval_loss_acc / args.eval_batchs, lables_list, predicts_list
            )
            wdl_module.train()
