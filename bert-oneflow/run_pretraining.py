#!/usr/bin/python3
import argparse
import os
import time
from functools import partial
from typing import List
from compare_lazy_outputs import load_params_from_lazy
from utils.reporter import Reporter

import copy

import numpy as np
import oneflow as flow
from oneflow import nn

import sys

sys.path.append(".")
from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR

def ttol(tensor, pure_local=True):
    """ to local """
    if tensor.is_consistent:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local()

    return tensor


def tton(tensor, local_only=True):
    """ tensor to numpy """
    if tensor.is_consistent:
        if local_only:
            tensor = tensor.to_local().numpy()
        else:
            tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
    else:
        tensor = tensor.numpy()

    return tensor


def save_model(module: nn.Module, checkpoint_path: str, epoch: int, acc: float):
    flow.save(
        module.state_dict(),
        os.path.join(checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, acc)),
    )


def train(epoch, iter_per_epoch, graph, print_interval, metric_local):
    total_loss = []
    total_mlm_loss = []
    total_ns_loss = []
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

        next_sent_output, next_sent_labels, loss, mlm_loss, ns_loss = graph()

        # to local
        next_sent_output = ttol(next_sent_output, metric_local)
        next_sent_labels = ttol(next_sent_labels, metric_local)
        loss = ttol(loss, metric_local)
        mlm_loss = ttol(mlm_loss, metric_local)
        ns_loss = ttol(ns_loss, metric_local)

        
        # Waiting for sync
        loss = loss.numpy().item()
        end_t = time.time()

        # next sentence prediction accuracy
        correct = (
            next_sent_output.argmax(dim=-1)
            .eq(next_sent_labels.squeeze(1))
            .sum()
            .numpy()
            .item()
        )
        total_loss.append(loss)
        total_mlm_loss.append(mlm_loss.numpy().item())
        total_ns_loss.append(ns_loss.numpy().item())
        total_correct += correct
        total_element += next_sent_labels.nelement()

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, train iter {}, loss {:.3f}, iter time: {:.3f}s".format(
                    epoch, (i + 1), np.mean(total_loss), end_t - start_t
                )
            )

    print(
        "Epoch {}, train iter {}, loss {:.3f}, total accuracy {:.2f}".format(
            epoch, (i + 1), np.mean(total_loss), total_correct * 100.0 / total_element
        )
    )
    return total_loss, total_mlm_loss, total_ns_loss


def validation(
    epoch: int, iter_per_epoch: int, graph: nn.Graph, print_interval: int
) -> float:
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

        next_sent_output, next_sent_labels = graph()

        next_sent_output = next_sent_output.numpy()
        next_sent_labels = next_sent_labels.numpy()
        end_t = time.time()

        # next sentence prediction accuracy
        correct = (
            next_sent_output.argmax(axis=-1) == next_sent_labels.squeeze(1)
        ).sum()
        total_correct += correct
        total_element += next_sent_labels.size

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, val iter {}, val time: {:.3f}s".format(
                    epoch, (i + 1), end_t - start_t
                )
            )

    print(
        "Epoch {}, val iter {}, total accuracy {:.2f}".format(
            epoch, (i + 1), total_correct * 100.0 / total_element
        )
    )
    return total_correct / total_element


def main():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="/dataset/bert_regression_test/0",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Validation batch size"
    )

    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Number of attention heads",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="intermediate size of bert encoder",
    )
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_size_per_head", type=int, default=64)
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
    parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")

    parser.add_argument(
        "--with-cuda",
        type=bool,
        default=True,
        help="Training with CUDA: true, or false",
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )

    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--print_interval", type=int, default=10, help="Interval of printing"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to model saving",
    )
    parser.add_argument(
        "--use_fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
    )
    parser.add_argument(
        "--nccl-fusion-threshold-mb",
        type=int,
        default=16,
        dest="nccl_fusion_threshold_mb",
        help="NCCL fusion threshold megabytes, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--nccl-fusion-max-ops",
        type=int,
        default=24,
        dest="nccl_fusion_max_ops",
        help="Maximum number of ops of NCCL fusion, set to 0 to compatible with previous version of OneFlow.",
    )
    parser.add_argument(
        "--use_consistent",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use consistent",
    )
    parser.add_argument(
        "--metric-local",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        dest="metric_local",
    )

    args = parser.parse_args()

    flow.boxing.nccl.set_fusion_threshold_mbytes(args.nccl_fusion_threshold_mb)
    flow.boxing.nccl.set_fusion_max_ops_num(args.nccl_fusion_max_ops)

    if args.with_cuda:
        device = "cuda"
    else:
        device = "cpu"

    print("Device is: ", device)

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.train_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=args.use_consistent,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=args.use_consistent,
    )

    print("Building BERT Model")
    bert_model = BertForPreTraining(
        args.vocab_size,
        args.seq_length,
        args.hidden_size,
        args.num_hidden_layers,
        args.num_attention_heads,
        args.intermediate_size,
        nn.GELU(),
        0.0,  # args.hidden_dropout_prob,
        0.0,  # args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )

    # Load the same initial parameters with lazy model.
    load_params_from_lazy(
        bert_model.state_dict(), "../../OneFlow-Benchmark/LanguageModeling/BERT/initial_model",
    )

    ns_criterion = nn.CrossEntropyLoss(reduction="mean")
    mlm_criterion = nn.CrossEntropyLoss(reduction="none")

    if args.use_consistent:
        placement = flow.placement("cuda", {0: range(flow.env.get_world_size())})
        bert_model = bert_model.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        ns_criterion = ns_criterion.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
        mlm_criterion = mlm_criterion.to_consistent(placement=placement, sbp=flow.sbp.broadcast)
    else:
        bert_model.to(device)
        ns_criterion.to(device)
        mlm_criterion.to(device)

    def build_adamW_optimizer(
        model: nn.Module,
        lr: float,
        weight_decay: float,
        weight_decay_excludes: List[str],
    ):
        defaults = {"lr": lr, "weight_decay": weight_decay, "epsilon": 1e-6}
        params = []
        for module_param_name, value in model.named_parameters():
            if not value.requires_grad:
                continue

            hyperparameters = copy.copy(defaults)
            for exclude_name in weight_decay_excludes:
                if module_param_name.find(exclude_name) != -1:
                    hyperparameters["weight_decay"] = 0
                    break

            params.append({"params": [value], **hyperparameters})

        return flow.optim.AdamW(params)
    
    def build_sgd_optimizer(
        model: nn.Module,
        lr: float,
        momentum: float,
    ):
        defaults = {"lr": lr, "momentum": momentum}
        params = []
        for module_param_name, value in model.named_parameters():
            if not value.requires_grad:
                continue
            hyperparameters = copy.copy(defaults)
            params.append({"params": [value], **hyperparameters})

        return flow.optim.SGD(params)

    # optimizer = build_adamW_optimizer(
    #     bert_model,
    #     args.lr,
    #     args.weight_decay,
    #     weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
    # )
    optimizer = build_sgd_optimizer(
        model=bert_model,
        lr=args.lr,
        momentum=0.9
    )

    steps = args.epochs * len(train_data_loader)

    lr_scheduler = PolynomialLR(optimizer, steps=100, end_learning_rate=0.0)

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler, warmup_factor=0, warmup_iters=50, warmup_method="linear"
    )

    def get_masked_lm_loss(
        logit_blob,
        masked_lm_positions,
        masked_lm_labels,
        label_weights,
        max_predictions_per_seq,
    ):
        # gather valid position indices
        logit_blob = flow.gather(
            logit_blob,
            index=masked_lm_positions.unsqueeze(2).repeat(1, 1, args.vocab_size),
            dim=1,
        )
        logit_blob = flow.reshape(logit_blob, [-1, args.vocab_size])
        label_id_blob = flow.reshape(masked_lm_labels, [-1])

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        pre_example_loss = mlm_criterion(logit_blob, label_id_blob)
        pre_example_loss = flow.reshape(pre_example_loss, [-1, max_predictions_per_seq])
        sum_label_weight = flow.sum(label_weights, dim=-1)
        sum_label_weight = sum_label_weight / label_weights.shape[0]
        numerator = flow.sum(pre_example_loss * label_weights)
        denominator = flow.sum(label_weights) + 1e-5
        loss = numerator / denominator
        return loss

    class BertGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self.ns_criterion = ns_criterion
            self.masked_lm_criterion = partial(
                get_masked_lm_loss, max_predictions_per_seq=args.max_predictions_per_seq
            )
            self.add_optimizer(optimizer, lr_sch=lr_scheduler)
            self._train_data_loader = train_data_loader
            if args.use_fp16:
                self.config.enable_amp(True)
                grad_scaler = flow.amp.GradScaler(
                    init_scale=2 ** 30,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
                self.set_grad_scaler(grad_scaler)
            self.config.allow_fuse_add_to_output(True)
            self.config.allow_fuse_model_update_ops(True)

        def build(self):

            (
                input_ids,
                next_sentence_labels,
                input_mask,
                segment_ids,
                masked_lm_ids,
                masked_lm_positions,
                masked_lm_weights,
            ) = self._train_data_loader()
            if not args.use_consistent:
                input_ids = input_ids.to(device=device)
                input_mask = input_mask.to(device=device)
                segment_ids = segment_ids.to(device=device)
                next_sentence_labels = next_sentence_labels.to(device=device)
                masked_lm_ids = masked_lm_ids.to(device=device)
                masked_lm_positions = masked_lm_positions.to(device=device)
                masked_lm_weights = masked_lm_weights.to(device=device)

            # 1. forward the next_sentence_prediction and masked_lm model
            prediction_scores, seq_relationship_scores = self.bert(
                input_ids, segment_ids, input_mask
            )

            # 2-1. loss of is_next classification result
            next_sentence_loss = self.ns_criterion(
                seq_relationship_scores.view(-1, 2), next_sentence_labels.view(-1)
            )

            masked_lm_loss = self.masked_lm_criterion(
                prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
            )

            total_loss = next_sentence_loss + masked_lm_loss

            total_loss.backward()
            return (
                seq_relationship_scores,
                next_sentence_labels,
                total_loss,
                masked_lm_loss,
                next_sentence_loss,
            )

    bert_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self._test_data_loader = test_data_loader
            if args.use_fp16:
                self.config.enable_amp(True)
            self.config.allow_fuse_add_to_output(True)

        def build(self):
            (
                input_ids,
                next_sent_labels,
                input_masks,
                segment_ids,
                masked_lm_ids,
                masked_lm_positions,
                masked_lm_weights,
            ) = self._test_data_loader()
            input_ids = input_ids.to(device=device)
            input_masks = input_masks.to(device=device)
            segment_ids = segment_ids.to(device=device)
            next_sent_labels = next_sent_labels.to(device=device)
            masked_lm_ids = masked_lm_ids.to(device=device)
            masked_lm_positions = masked_lm_positions.to(device)

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                _, seq_relationship_scores = self.bert(
                    input_ids, input_masks, segment_ids
                )

            return seq_relationship_scores, next_sent_labels

    bert_eval_graph = BertEvalGraph()

    train_total_losses = []
    train_lml_losses = []
    train_ns_losses = []

    for epoch in range(args.epochs):
        # Train
        bert_model.train()
        train_total_loss, lml_loss, ns_loss = train(
            epoch, 100, bert_graph, args.print_interval, args.metric_local  # len(train_data_loader),
        )

        train_total_losses.extend(train_total_loss)
        train_lml_losses.extend(lml_loss)
        train_ns_losses.extend(ns_loss)

    save_dir = "loss_txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Reporter.write2file(
        train_total_losses, os.path.join(save_dir, "bert_graph_sgd_amp_consistent_loss.txt")
    )
    # Reporter.write2file(
    #     train_lml_losses, os.path.join(save_dir, "bert_graph_lml_loss.txt")
    # )
    # Reporter.write2file(
    #     train_ns_losses, os.path.join(save_dir, "bert_graph_ns_loss.txt")
    # )
    # Eval
    # bert_model.eval()
    # val_acc = validation(
    #     epoch, len(test_data_loader), bert_eval_graph, args.print_interval * 10
    # )

    # print("Saveing model ...")
    # save_model(bert_model, args.checkpoint_path, epoch, val_acc)


if __name__ == "__main__":
    main()
