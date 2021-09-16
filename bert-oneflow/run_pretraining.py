#!/usr/bin/python3
import argparse
import os
import time
from functools import partial
from typing import Dict
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
from utils.optimizer import build_adamW_optimizer, build_sgd_optimizer
from utils.metric import Metric


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


def pretrain(graph: nn.Graph, metric_local: bool) -> Dict:

    # NOTE(lxy): when using gradient accumulation, graph call 1 step for 1 mini-batch(n micro-batch)
    next_sent_output, next_sent_labels, loss, mlm_loss, nsp_loss = graph()

    # to local
    next_sent_output = ttol(next_sent_output, metric_local)
    next_sent_labels = ttol(next_sent_labels, metric_local)

    # next sentence prediction accuracy
    correct = (
        next_sent_output.argmax(dim=-1)
        .to(dtype=next_sent_labels.dtype)
        .eq(next_sent_labels.squeeze(1))
        .to(dtype=flow.float32)
        .sum()
        .numpy()
        .item()
    )
    pred_acc = np.array(correct / next_sent_labels.nelement())

    return {
        "total_loss": tton(loss.mean(), False),
        "mlm_loss": tton(mlm_loss.mean(), metric_local),
        "nsp_loss": tton(nsp_loss.mean(), metric_local),
        "pred_acc": pred_acc,
    }


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
        # default="/dataset/bert_regression_test/0",
        default="wiki_ofrecord_seq_len_128_example",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=32, help="Validation batch size"
    )
    parser.add_argument(
        "--train-global-batch-size",
        type=int,
        default=None,
        dest="train_global_batch_size",
        help="train batch size",
    )
    parser.add_argument(
        "--val-global-batch-size",
        type=int,
        default=None,
        dest="val_global_batch_size",
        help="val batch size",
    )

    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Number of attention heads",
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

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--loss_print_every_n_iters", type=int, default=20, help="Interval of printing"
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
        "--use-grad-acc",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use gradient accumulation"
    )
    parser.add_argument("--grad-acc-steps", type=int, default=1, help="Steps for gradient accumulation")
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
        default=False,
        nargs="?",
        const=True,
        dest="metric_local",
    )

    args = parser.parse_args()

    world_size = flow.env.get_world_size()
    if args.train_global_batch_size is None:
        args.train_global_batch_size = args.train_batch_size * world_size
    else:
        assert args.train_global_batch_size % args.train_batch_size == 0

    if args.val_global_batch_size is None:
        args.val_global_batch_size = args.val_batch_size * world_size
    else:
        assert args.val_global_batch_size % args.val_batch_size == 0

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
        batch_size=args.train_global_batch_size,
        data_part_num=4,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=args.use_consistent,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_global_batch_size,
        data_part_num=4,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=args.use_consistent,
    )

    print("Building BERT Model")
    hidden_size = 64 * args.num_attention_heads
    intermediate_size = 4 * hidden_size
    bert_model = BertForPreTraining(
        args.vocab_size,
        args.seq_length,
        hidden_size,
        args.num_hidden_layers,
        args.num_attention_heads,
        intermediate_size,
        nn.GELU(),
        0.0,  # args.hidden_dropout_prob,
        0.0,  # args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )

    # Load the same initial parameters with lazy model.
    # load_params_from_lazy(
    #     bert_model.state_dict(),
    #     "../../OneFlow-Benchmark/LanguageModeling/BERT/initial_model",
    # )
    assert id(bert_model.cls.predictions.decoder.weight) == id(
        bert_model.bert.embeddings.word_embeddings.weight
    )

    ns_criterion = nn.CrossEntropyLoss(reduction="mean")
    mlm_criterion = nn.CrossEntropyLoss(reduction="none")

    if args.use_consistent:
        placement = flow.placement("cuda", {0: range(flow.env.get_world_size())})
        bert_model = bert_model.to_consistent(
            placement=placement, sbp=flow.sbp.broadcast
        )
    else:
        bert_model.to(device)
        ns_criterion.to(device)
        mlm_criterion.to(device)

    optimizer = build_sgd_optimizer(  # build_adamW_optimizer(
        bert_model,
        args.lr,
        momentum=0.9,
        clip_grad_max_norm=1.0,
        clip_grad_norm_type=2.0
        # args.weight_decay,
        # weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
    )

    steps = args.epochs * len(train_data_loader)

    lr_scheduler = PolynomialLR(optimizer, steps=300, end_learning_rate=0.0)

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
        # NOTE(lxy): `repeat` and `expand` will convert `logit_blob` sbp from S(0) to B
        # logit_blob = flow.gather(
        #     logit_blob,
        #     index=masked_lm_positions.unsqueeze(2).repeat(1, 1, args.vocab_size),
        #     dim=1,
        # )
        if logit_blob.is_consistent:
            zeros = flow.zeros(
                (1, 1, args.vocab_size),
                dtype=masked_lm_positions.dtype,
                placement=masked_lm_positions.placement,
                sbp=flow.sbp.broadcast,
            )
        else:
            zeros = flow.zeros((1, 1, args.vocab_size), dtype=masked_lm_positions.dtype, device=masked_lm_positions.device)
        masked_lm_positions = masked_lm_positions.unsqueeze(2) + zeros

        # gather valid position indices
        logit_blob = flow.gather(logit_blob, index=masked_lm_positions, dim=1,)

        logit_blob = flow.reshape(logit_blob, [-1, args.vocab_size])
        label_id_blob = flow.reshape(masked_lm_labels, [-1])

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        pre_example_loss = mlm_criterion(logit_blob, label_id_blob)
        pre_example_loss = flow.reshape(pre_example_loss, [-1, max_predictions_per_seq])
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
            if args.use_grad_acc:
                self.config.set_gradient_accumulation_steps(args.grad_acc_steps)
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
                seq_relationship_scores.reshape(-1, 2), next_sentence_labels.reshape(-1)
            )

            masked_lm_loss = self.masked_lm_criterion(
                prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
            )

            total_loss = masked_lm_loss + next_sentence_loss

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

    for epoch in range(args.epochs):
        metric = Metric(
            desc="bert pretrain",
            print_steps=args.loss_print_every_n_iters,
            batch_size=args.train_batch_size,
            keys=["total_loss", "mlm_loss", "nsp_loss", "pred_acc"],
        )

        # Train
        bert_model.train()

        for step in range(1000):  # range(len(train_data_loader)):
            bert_outputs = pretrain(bert_graph, args.metric_local)

            if (flow.env.get_rank() == 0):
                metric.metric_cb(step, epoch=epoch)(bert_outputs)

            train_total_losses.append(bert_outputs["total_loss"])

    save_dir = "loss_txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Reporter.write2file(
        train_total_losses,
        os.path.join(save_dir, "bert_graph_sgd_amp_consistent_ddp_4gpu_4partdiff_clip_loss.txt"),
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
    # save_model(bert_model, args.checkpoint_path, epoch, 0.1)


if __name__ == "__main__":
    main()
