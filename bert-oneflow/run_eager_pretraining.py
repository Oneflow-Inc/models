#!/usr/bin/python3
import argparse
import os
import time
from functools import partial
from compare_lazy_outputs import load_params_from_lazy

import numpy as np
import oneflow as flow
from oneflow import nn

from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.reporter import Reporter


def save_model(module: nn.Module, checkpoint_path: str, epoch: int, acc: float):
    flow.save(
        module.state_dict(),
        os.path.join(checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, acc)),
    )


def train(
    epoch,
    iter_per_epoch,
    data_loader,
    model,
    ns_criterion,
    masked_lm_criterion,
    optimizer,
    print_interval,
    device="cuda",
):
    total_losses = []
    total_mlm_loss = []
    total_ns_loss = []
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

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
        prediction_scores, seq_relationship_scores = model(
            input_ids, segment_ids, input_mask
        )

        # 2-1. loss of is_next classification result
        next_sentence_loss = ns_criterion(
            seq_relationship_scores.view(-1, 2), next_sentence_labels.view(-1)
        )

        masked_lm_loss = masked_lm_criterion(
            prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
        )

        total_loss = next_sentence_loss + masked_lm_loss

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # next sentence prediction accuracy
        correct = (
            seq_relationship_scores.argmax(dim=-1)
            .eq(next_sentence_labels.squeeze(1))
            .sum()
            .numpy()
            .item()
        )
        # Waiting for sync
        end_t = time.time()

        total_losses.append(total_loss.numpy().item())
        total_mlm_loss.append(masked_lm_loss.numpy().item())
        total_ns_loss.append(next_sentence_loss.numpy().item())
        total_correct += correct
        total_element += next_sentence_labels.nelement()

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, train iter {}, loss {:.3f}, iter time: {:.3f}s".format(
                    epoch, (i + 1), np.mean(total_losses), end_t - start_t
                )
            )

    print(
        "Epoch {}, train iter {}, loss {:.3f}, total accuracy {:.2f}".format(
            epoch, (i + 1), np.mean(total_losses), total_correct * 100.0 / total_element
        )
    )
    return total_losses, total_mlm_loss, total_ns_loss


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ofrecord-path",
        type=str,
        default="wiki_ofrecord_seq_len_128_example",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=32, help="Validation batch size"
    )

    parser.add_argument(
        "--hidden-size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--hidden-layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a", "--atten-heads", type=int, default=12, help="Number of attention heads"
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=3072,
        help="intermediate size of bert encoder",
    )
    parser.add_argument("--max-position-embeddings", type=int, default=512)
    parser.add_argument(
        "-s", "--seq-length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--type-vocab-size", type=int, default=2)
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.1)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.1)
    parser.add_argument("--hidden-size-per-head", type=int, default=64)
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

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
    parser.add_argument(
        "--adam-weight-decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9, help="Adam first beta value"
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999, help="Adam first beta value"
    )
    parser.add_argument(
        "--print-interval", type=int, default=10, help="Interval of printing"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints",
        help="Path to model saving",
    )

    args = parser.parse_args()

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.train_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=20,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=20,
    )

    print("Building BERT Model")
    bert_model = BertForPreTraining(
        args.vocab_size,
        args.seq_length,
        args.hidden_size,
        args.hidden_layers,
        args.atten_heads,
        args.intermediate_size,
        nn.GELU(),
        0.0,  # args.hidden_dropout_prob,
        0.0,  # args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )
    # print(bert_model)

    # Load the same initial parameters with lazy model.
    load_params_from_lazy(
        bert_model.state_dict(),
        flow.load(
            "../../OneFlow-Benchmark/LanguageModeling/BERT/snapshots/snapshot_snapshot_1"
        ),
    )

    bert_model.to(device)

    # optimizer = flow.optim.Adam(
    #     bert_model.parameters(),
    #     lr=args.lr,
    #     # weight_decay=args.adam_weight_decay,
    #     betas=(args.adam_beta1, args.adam_beta2),
    # )
    optimizer = flow.optim.SGD(bert_model.parameters(), lr=1e-4, momentum=0.9)

    steps = args.epochs * len(train_data_loader)
    cosine_annealing_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps
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
    train_mlm_losses = []
    train_nsp_losses = []
    for epoch in range(args.epochs):
        # Train
        bert_model.train()
        train_total_loss, mlm_loss, nsp_loss = train(
            epoch,
            100,
            train_data_loader,
            bert_model,
            ns_criterion,
            partial(get_masked_lm_loss, max_prediction_per_seq=20),
            optimizer,
            args.print_interval,
        )

        train_total_losses.extend(train_total_loss)
        train_mlm_losses.extend(mlm_loss)
        train_nsp_losses.extend(nsp_loss)

    save_dir = "temp1"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Reporter.write2file(train_total_losses, os.path.join(save_dir, "eager_loss.txt"))
    Reporter.write2file(train_mlm_losses, os.path.join(save_dir, "eager_lml_loss.txt"))
    Reporter.write2file(train_nsp_losses, os.path.join(save_dir, "eager_nsp_loss.txt"))


if __name__ == "__main__":
    main()
