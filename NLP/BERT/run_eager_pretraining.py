#!/usr/bin/python3
import time
from functools import partial

from run_pretraining import get_config
import numpy as np
import oneflow as flow
from oneflow.nn.parallel import DistributedDataParallel as ddp
from oneflow import nn

from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR
from utils.optimizer import build_optimizer
from utils.metric import Metric
from utils.comm import ttol, tton
from utils.checkpoint import save_model


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
    prediction_scores, seq_relationship_scores = model(
        input_ids, segment_ids, input_mask
    )

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
        "total_loss": tton(total_loss, False),
        "mlm_loss": tton(masked_lm_loss),
        "nsp_loss": tton(next_sentence_loss),
        "pred_acc": pred_acc,
    }


def validation(
    epoch: int,
    data_loader: OfRecordDataLoader,
    model: nn.Module,
    print_interval: int,
    device="cuda",
) -> float:
    total_correct = 0
    total_element = 0
    for i in range(len(data_loader)):

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

        start_t = time.time()
        with flow.no_grad():
            _, next_sentence_output = model(input_ids, segment_ids, input_mask)

        next_sentence_output = tton(next_sentence_output)
        next_sent_labels = tton(next_sentence_labels)
        end_t = time.time()

        # next sentence prediction accuracy
        correct = (
            next_sentence_output.argmax(axis=-1) == next_sent_labels.squeeze(1)
        ).sum()
        total_correct += correct
        total_element += next_sent_labels.size

        if (i + 1) % print_interval == 0 and flow.env.get_rank() == 0:
            print(
                "Epoch {}, val iter {}, val time: {:.3f}s".format(
                    epoch, (i + 1), end_t - start_t
                )
            )

    if flow.env.get_rank() == 0:
        print(
            "Epoch {}, val iter {}, total accuracy {:.2f}".format(
                epoch, (i + 1), total_correct * 100.0 / total_element
            )
        )
    return total_correct / total_element


def main():

    args = get_config()

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="train",
        dataset_size=args.train_dataset_size,
        batch_size=args.train_batch_size,
        data_part_num=args.train_data_part,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=False,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=4,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=False,
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
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )

    # Load the same initial parameters with lazy model.
    # from utils.compare_lazy_outputs import load_params_from_lazy
    # load_params_from_lazy(
    #     bert_model.state_dict(),
    #     "../../OneFlow-Benchmark/LanguageModeling/BERT/initial_model",
    # )

    bert_model = bert_model.to(device)
    if args.use_ddp:
        bert_model = ddp(bert_model)

    optimizer = build_optimizer(
        args.optim_name,
        bert_model,
        args.lr,
        args.weight_decay,
        weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
        clip_grad_max_norm=1,
        clip_grad_norm_type=2.0,
    )

    steps = args.epochs * len(train_data_loader)
    warmup_steps = int(steps * args.warmup_proportion)

    lr_scheduler = PolynomialLR(optimizer, steps=steps, end_learning_rate=0.0)

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler, warmup_factor=0, warmup_iters=warmup_steps, warmup_method="linear"
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

        for step in range(len(train_data_loader)):
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

            if flow.env.get_rank() == 0:
                metric.metric_cb(step, epoch=epoch)(bert_outputs)

            train_total_losses.append(bert_outputs["total_loss"])

        # Eval
        bert_model.eval()
        val_acc = validation(
            epoch, test_data_loader, bert_model, args.val_print_every_n_iters
        )

        save_model(bert_model, args.checkpoint_path, epoch, val_acc, False)


if __name__ == "__main__":
    main()
