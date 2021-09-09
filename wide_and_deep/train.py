import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import oneflow as flow
from tqdm import tqdm
from config import get_args
from models.dataloader_utils import OFRecordDataLoader
from oneflow.framework import distribute
from models.wide_and_deep_module import WideAndDeep
from util import dump_to_npy, save_param_npy
from oneflow.nn.parallel import DistributedDataParallel as ddp
from graph import WideAndDeepGraph,WideAndDeepTrainGraph


world_size = flow.env.get_world_size()
placement = flow.placement("cpu", {0: range(world_size)})

def prepare_modules(args):
    world_size = flow.env.get_world_size()
    placement = flow.placement("cpu", {0: range(world_size)})

    graph_or_eager=args.graph_or_eager #'graph' or 'eager'
    mode=args.mode #'ddp' or 'dmp'
    is_consistent=args.is_consistent

    wdl_module = WideAndDeep(args)
    #dmp的，写模型的代码里面已经转好为consistent了
    if graph_or_eager=='eager' and mode=='ddp' and is_consistent==False:
        #对标pytorch的ddp
        placement = None
        sbp = None
    elif graph_or_eager=='graph' and mode=='ddp' and is_consistent==True:
        #graph_train_consistent
        sbp = flow.sbp.split(0)
        #model->consistent
        wdl_module = wdl_module.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
    elif graph_or_eager=='graph' and mode=='dmp' and is_consistent==True:
        #graph_train_consistent_dmp
        sbp = flow.sbp.broadcast

    elif graph_or_eager=='eager' and mode=='ddp' and is_consistent==True:
        #eager_train_consistent
        sbp = flow.sbp.split(0)
        wdl_module = wdl_module.to_consistent(placement=placement, sbp=flow.sbp.broadcast)

    elif graph_or_eager=='eager' and mode=='dmp' and is_consistent==True:
        #eager_train_consistent_dmp
        sbp = flow.sbp.broadcast

    else:
        print('不支持的训练模式')
        return
    train_dataloader = OFRecordDataLoader(args, data_root=args.data_dir, batch_size=args.batch_size,placement=placement,sbp=sbp)
    val_dataloader = OFRecordDataLoader(args, data_root=args.data_dir, mode="val",batch_size=args.batch_size,placement=placement,sbp=sbp)
    wdl_module.to("cuda")
    bce_loss = flow.nn.BCELoss(reduction="mean")
    bce_loss.to("cuda")
    opt = flow.optim.SGD(wdl_module.parameters(), lr=args.learning_rate, momentum=0.9)

    if args.model_load_dir != "":
        print("load checkpointed model from ", args.model_load_dir)
        wdl_module.load_state_dict(flow.load(args.model_load_dir))

    if args.save_initial_model and args.model_save_dir != "":
        path = os.path.join(args.model_save_dir, "initial_checkpoint")
        if not os.path.isdir(path):
            flow.save(wdl_module.state_dict(), path)

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


if __name__ == '__main__':
    args = get_args()
    graph_or_eager=args.graph_or_eager #'graph' or 'eager'
    mode=args.mode #'ddp' or 'dmp'
    is_consistent=args.is_consistent

    train_dataloader, val_dataloader, wdl_module, loss, opt = prepare_modules(
        args)
    if graph_or_eager=='graph':
        eval_graph = WideAndDeepGraph(wdl_module,val_dataloader,loss)
        train_graph = WideAndDeepTrainGraph(wdl_module,train_dataloader,loss,opt)   
    losses = []
    wdl_module.train()
    for i in tqdm(range(args.max_iter)):
        if graph_or_eager=='graph':
            predicts, labels, train_loss = train_graph()
        else:
            labels, dense_fields, wide_sparse_fields, deep_sparse_fields = train_dataloader()
            labels = labels.to("cuda").to(dtype=flow.float32)
            dense_fields = dense_fields.to("cuda")
            wide_sparse_fields = wide_sparse_fields.to("cuda")
            deep_sparse_fields = deep_sparse_fields.to("cuda")
            predicts = wdl_module(
                dense_fields, wide_sparse_fields, deep_sparse_fields)
            train_loss = loss(predicts, labels)
        
        #train_loss是partial_sum，比如consistent模式下的ddp
        if is_consistent==True and train_loss.sbp==flow.sbp.partial_sum:
            train_loss = train_loss / world_size
        #各个rank 打印local loss
        if is_consistent==True:
            losses.append(train_loss.to_local().numpy().mean())
        else:
            losses.append(train_loss.numpy().mean())
            train_loss.backward()
            opt.step()
            opt.zero_grad()
        if (i+1) % args.print_interval == 0:
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
                if graph_or_eager=='graph':
                    predicts, labels, eval_loss = eval_graph()
                else:
                    labels, dense_fields, wide_sparse_fields, deep_sparse_fields = val_dataloader()
                    labels = labels.to(dtype=flow.float32).to("cuda")
                    dense_fields = dense_fields.to("cuda")
                    wide_sparse_fields = wide_sparse_fields.to("cuda")
                    deep_sparse_fields = deep_sparse_fields.to("cuda")
                    predicts = wdl_module(
                        dense_fields, wide_sparse_fields, deep_sparse_fields)
                    eval_loss = loss(predicts, labels)
                
                if is_consistent==True and eval_loss.sbp==flow.sbp.partial_sum:
                    eval_loss = eval_loss / world_size
                #各个rank 打印local loss
                if is_consistent==True:
                    eval_loss_acc += eval_loss.to_local().numpy().mean()
                    lables_list.append(labels.to_local().numpy())
                    predicts_list.append(predicts.to_local().numpy())
                else:
                    eval_loss_acc += eval_loss.numpy().mean()
                    lables_list.append(labels.numpy())
                    predicts_list.append(predicts.numpy())
            print_eval_metrics(i+1, eval_loss_acc/args.eval_batchs,
                               lables_list, predicts_list)
            wdl_module.train()
