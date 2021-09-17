import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import oneflow as flow

from config import get_args
from dataloader_utils import OFRecordDataLoader
from wide_and_deep_module import WideAndDeep
from util import dump_to_npy, save_param_npy


if __name__ == '__main__':
    # flow.InitEagerGlobalSession()

    args = get_args()

    train_dataloader = OFRecordDataLoader(args, data_root=args.data_dir)
    val_dataloader = OFRecordDataLoader(
        args, data_root=args.data_dir, mode="val")
    wdl_module = WideAndDeep(args)
    if args.model_load_dir != "":
        print("load checkpointed model from ", args.model_load_dir)
        wdl_module.load_state_dict(flow.load(args.model_load_dir))

    if args.save_initial_model and args.model_save_dir != "":
        path = os.path.join(args.model_save_dir, 'initial_checkpoint')
        if not os.path.isdir(path):
            flow.save(wdl_module.state_dict(), path)
    # save_param_npy(wdl_module)

    bce_loss = flow.nn.BCELoss(reduction="mean")

    wdl_module.to("cuda")
    bce_loss.to("cuda")

    # opt = flow.optim.Adam(
    #     wdl_module.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
    #     # do_bias_correction=True
    # )
    opt = flow.optim.SGD(
        wdl_module.parameters(), lr=args.learning_rate, momentum=0.0,
    )
    losses = []
    wdl_module.train()
    for i in range(args.max_iter):
        labels, dense_fields, wide_sparse_fields, deep_sparse_fields = train_dataloader()
        #dump_to_npy(labels, sub=i)
        #dump_to_npy(dense_fields, sub=i)
        #dump_to_npy(wide_sparse_fields, sub=i)
        #dump_to_npy(deep_sparse_fields, sub=i)
        labels = labels.to("cuda").to(dtype=flow.float32)
        dense_fields = dense_fields.to("cuda")
        wide_sparse_fields = wide_sparse_fields.to("cuda")
        deep_sparse_fields = deep_sparse_fields.to("cuda")
        predicts, deep_weight = wdl_module(
            dense_fields, wide_sparse_fields, deep_sparse_fields)
        predicts.retain_grad()
        deep_weight.retain_grad()
        loss = bce_loss(predicts, labels)
        loss.retain_grad()

        #dump_to_npy(predicts, sub=i)
        #dump_to_npy(loss, sub=i)
        losses.append(loss.numpy().mean())
        # loss.backward(flow.ones_like(loss))
        loss.backward()
        # print(predicts.grad)
        # print(loss.grad)
        # dump_to_npy(deep_weight, name=f'{i}/deep_weight')
        # dump_to_npy(deep_weight.grad, name=f'{i}/deep_weight_grad')
        # dump_to_npy(predicts.grad, name=f'{i}/predicts_grad')
        # dump_to_npy(loss.grad, name=f'{i}/loss_grad')
        opt.step()
        opt.zero_grad()
        # time.sleep(0.1)
        # print(deep_sparse_fields)
        # print(loss)
        if (i+1) % args.print_interval == 0:
            l = sum(losses) / len(losses)
            losses = []
            print(f"iter {i} train_loss {l} time {time.time()}")
            if args.eval_batchs <= 0:
                continue

            eval_loss = 0.0
            wdl_module.eval()
            lables_list = []
            predicts_list = []
            for j in range(args.eval_batchs):
                labels, dense_fields, wide_sparse_fields, deep_sparse_fields = val_dataloader()
                labels = labels.to("cuda").to(dtype=flow.float32)
                # print(labels.numpy().flatten())
                # print(j)
                dense_fields = dense_fields.to("cuda")
                wide_sparse_fields = wide_sparse_fields.to("cuda")
                deep_sparse_fields = deep_sparse_fields.to("cuda")
                predicts = wdl_module(
                    dense_fields, wide_sparse_fields, deep_sparse_fields)
                loss = bce_loss(predicts, labels)
                eval_loss += loss.numpy().mean()
                lables_list.append(labels.numpy())
                predicts_list.append(predicts.numpy())
            all_labels = np.concatenate(lables_list, axis=0)
            all_predicts = np.concatenate(predicts_list, axis=0)
            # print(all_labels.shape, all_predicts.shape)
            # print(np.isnan(all_predicts).any())
            # print(all_labels.flatten())
            auc = "NaN" if np.isnan(all_predicts).any(
            ) else roc_auc_score(all_labels, all_predicts)
            print(f"iter {i} eval_loss {eval_loss/args.eval_batchs} auc {auc}")

            wdl_module.train()
    # flow.save(wdl_module.state_dict(), "output/iter0_checkpoint")
