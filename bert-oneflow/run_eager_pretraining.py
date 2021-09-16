#!/usr/bin/python3
import argparse
import os
import time
from functools import partial
from compare_lazy_outputs import load_params_from_lazy

import numpy as np
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow import nn

from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR
from utils.optimizer import build_adamW_optimizer, build_sgd_optimizer
from utils.reporter import Reporter
from utils.metric import Metric
from run_pretraining import ttol, tton


def save_model(module: nn.Module, checkpoint_path: str, epoch: int, acc: float):
    flow.save(
        module.state_dict(),
        os.path.join(checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, acc)),
    )


def pretrain(
    data_loader,
    model,
    ns_criterion,
    masked_lm_criterion,
    optimizer,
    lr_scheduler,
    device="cuda",
):

    (
        input_ids,
        next_sentence_labels,
        input_mask,
        segment_ids,
        masked_lm_ids,
        masked_lm_positions,
        masked_lm_weights,
    ) = data_loader()
    input_ids = input_ids.to(device=device)
    input_mask = input_mask.to(device=device)
    segment_ids = segment_ids.to(device=device)
    next_sentence_labels = next_sentence_labels.to(device=device)
    masked_lm_ids = masked_lm_ids.to(device=device)
    masked_lm_positions = masked_lm_positions.to(device=device)
    masked_lm_weights = masked_lm_weights.to(device=device)

    # 1. forward the next_sentence_prediction and masked_lm model
    prediction_scores, seq_relationship_scores = model(input_ids, segment_ids, input_mask)

    # 2-1. loss of is_next classification result
    next_sentence_loss = ns_criterion(
        seq_relationship_scores.reshape(-1, 2), next_sentence_labels.reshape(-1)
    )

    masked_lm_loss = masked_lm_criterion(
        prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
    )

    total_loss = next_sentence_loss + masked_lm_loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    lr_scheduler.step()

    seq_relationship_scores = ttol(seq_relationship_scores)
    next_sentence_labels = ttol(next_sentence_labels)
    # next sentence prediction accuracy
    correct = (
        seq_relationship_scores.argmax(dim=-1)
        .to(dtype=next_sentence_labels.dtype)
        .eq(next_sentence_labels.squeeze(1))
        .to(dtype=flow.float32)
        .sum()
        .numpy()
        .item()
    )

    pred_acc = np.array(correct / next_sentence_labels.nelement())

    return {
        "total_loss": tton(total_loss,False),
        "mlm_loss": tton(masked_lm_loss),
        "nsp_loss": tton(next_sentence_loss),
        "pred_acc": pred_acc,
    }


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
        "--use_ddp",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
    )

    args = parser.parse_args()

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.train_batch_size,
        data_part_num=4,
        seq_length=args.seq_length,
        max_predictions_per_seq=20,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=4,
        seq_length=args.seq_length,
        max_predictions_per_seq=20,
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
        bert_model.state_dict(),
        "../../OneFlow-Benchmark/LanguageModeling/BERT/initial_model",
    )

    bert_model = bert_model.to(device)
    if args.use_ddp:
        bert_model = ddp(bert_model)

    optimizer = build_sgd_optimizer(
        bert_model,
        args.lr,
        momentum=0.9
    )

    steps = args.epochs * len(train_data_loader)

    lr_scheduler = PolynomialLR(optimizer, steps=300, end_learning_rate=0.0)

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler, warmup_factor=0, warmup_iters=50, warmup_method="linear"
    )

    ns_criterion = nn.CrossEntropyLoss(reduction="mean")
    mlm_criterion = nn.CrossEntropyLoss(reduction="none")

    def get_masked_lm_loss(
        logit_blob,
        masked_lm_positions,
        masked_lm_labels,
        label_weights,
        max_prediction_per_seq,
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
        pre_example_loss = flow.reshape(pre_example_loss, [-1, max_prediction_per_seq])
        sum_label_weight = flow.sum(label_weights, dim=-1)
        sum_label_weight = sum_label_weight / label_weights.shape[0]
        numerator = flow.sum(pre_example_loss * label_weights)
        denominator = flow.sum(label_weights) + 1e-5
        loss = numerator / denominator
        return loss

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

        for step in range(300):
            bert_outputs = pretrain(
                train_data_loader,
                bert_model,
                ns_criterion,
                partial(
                    get_masked_lm_loss,
                    max_prediction_per_seq=args.max_predictions_per_seq,
                ),
                optimizer,
                lr_scheduler,
            )

            if (flow.env.get_rank() == 0):
                metric.metric_cb(step, epoch=epoch)(bert_outputs)

            train_total_losses.append(bert_outputs["total_loss"])

    save_dir = "loss_txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    Reporter.write2file(
        train_total_losses, os.path.join(save_dir, f"bert_4gpu_eager_diff_loss{flow.env.get_rank()}.txt")
    )


if __name__ == "__main__":
    main()
