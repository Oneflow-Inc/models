#!/usr/bin/python3
import argparse
import os
import time
from functools import partial

import numpy as np
import oneflow as flow
from oneflow import nn

from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader


def save_model(module: nn.Module, checkpoint_path: str, epoch: int, acc: float):
    flow.save(
        module.state_dict(),
        os.path.join(checkpoint_path, "epoch_%d_val_acc_%f" % (epoch, acc)),
    )


def train(epoch, iter_per_epoch, graph, print_interval):
    total_loss = 0
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

        next_sent_output, next_sent_labels, loss = graph()

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
        total_loss += loss
        total_correct += correct
        total_element += next_sent_labels.nelement()

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, train iter {}, loss {:.3f}, iter time: {:.3f}s".format(
                    epoch, (i + 1), total_loss / (i + 1), end_t - start_t
                )
            )

    print(
        "Epoch {}, train iter {}, loss {:.3f}, total accuracy {:.2f}".format(
            epoch, (i + 1), total_loss / (i + 1), total_correct * 100.0 / total_element
        )
    )


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

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="wiki_ofrecord_seq_len_128_example",
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
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
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
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")

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
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="Adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="Adam first beta value"
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

    args = parser.parse_args()

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="train",
        dataset_size=1024,
        batch_size=args.train_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=1,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
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
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )
    # print(bert_model)

    bert_model.to(device)

    optimizer = flow.optim.Adam(
        bert_model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2),
    )

    steps = args.epochs * len(train_data_loader)
    cosine_annealing_lr = flow.optim.lr_scheduler.CosineDecayLR(
        optimizer, decay_steps=steps
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

    class BertGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self.ns_criterion = ns_criterion
            self.masked_lm_criterion = partial(
                get_masked_lm_loss, max_prediction_per_seq=args.max_predictions_per_seq
            )
            self.add_optimizer(optimizer, lr_sch=cosine_annealing_lr)
            self._train_data_loader = train_data_loader

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
                seq_relationship_scores.view(-1, 2), next_sentence_labels.view(-1)
            )

            masked_lm_loss = self.masked_lm_criterion(
                prediction_scores, masked_lm_positions, masked_lm_ids, masked_lm_weights
            )

            total_loss = next_sentence_loss + masked_lm_loss

            total_loss.backward()
            return seq_relationship_scores, next_sentence_labels, total_loss

    bert_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self._test_data_loader = test_data_loader

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

    for epoch in range(args.epochs):
        # Train
        bert_model.train()
        train(epoch, len(train_data_loader), bert_graph, args.print_interval)

    # Eval
    bert_model.eval()
    val_acc = validation(
        epoch, len(test_data_loader), bert_eval_graph, args.print_interval * 10
    )

    print("Saveing model ...")
    save_model(bert_model, args.checkpoint_path, epoch, val_acc)


if __name__ == "__main__":
    main()
