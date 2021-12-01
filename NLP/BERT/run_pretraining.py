#!/usr/bin/python3
import time
from functools import partial
from typing import Dict


import numpy as np
import oneflow as flow
from oneflow import nn

import sys

sys.path.append(".")
from modeling import BertForPreTraining
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.lr_scheduler import PolynomialLR
from utils.optimizer import build_optimizer
from utils.metric import Metric
from utils.comm import ttol, tton
from utils.checkpoint import save_model

import config
from config import str2bool


def get_config():
    parser = config.get_parser()

    # pretrain bert config
    parser.add_argument(
        "--ofrecord_path",
        type=str,
        default="/dataset/bert/of_wiki_seq_len_128",
        help="Path to ofrecord dataset",
    )
    parser.add_argument(
        "--train-dataset-size",
        type=int,
        default=10000000,
        help="dataset size of ofrecord",
    )
    parser.add_argument(
        "--train-data-part", type=int, default=64, help="data part num of ofrecord"
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
    parser.add_argument(
        "--optim_name", type=str, default="adamw", help="optimizer name"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of adam")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--loss_print_every_n_iters",
        type=int,
        default=20,
        help="Interval of training loss printing",
    )
    parser.add_argument(
        "--val_print_every_n_iters",
        type=int,
        default=20,
        help="Interval of evaluation printing",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="Path to model saving",
    )
    parser.add_argument(
        "--grad-acc-steps", type=int, default=1, help="Steps for gradient accumulation"
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
        "--use_ddp",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
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
    return args


def pretrain(graph: nn.Graph, metric_local: bool) -> Dict:

    # NOTE(xyliao): when using gradient accumulation, graph call 1 step for 1 mini-batch(n micro-batch)
    next_sent_output, next_sent_labels, loss, mlm_loss, nsp_loss = graph()

    # to local
    # next_sent_output = ttol(next_sent_output, metric_local)
    # next_sent_labels = ttol(next_sent_labels, metric_local)

    # next sentence prediction accuracy
    # correct = (
    #     next_sent_output.argmax(dim=1)
    #     .to(dtype=next_sent_labels.dtype)
    #     .eq(next_sent_labels.squeeze(1))
    #     .to(dtype=flow.float32)
    #     .sum()
    #     .numpy()
    #     .item()
    # )
    # pred_acc = np.array(correct / next_sent_labels.nelement())

    return {
        # "total_loss": tton(loss.mean(), metric_local),
        # "mlm_loss": tton(mlm_loss.mean(), metric_local),
        # "nsp_loss": tton(nsp_loss.mean(), metric_local),
        # "pred_acc": pred_acc,
    }, loss


def validation(
    epoch: int,
    iter_per_epoch: int,
    graph: nn.Graph,
    print_interval: int,
    metric_local: bool,
) -> float:
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

        next_sent_output, next_sent_labels = graph()

        next_sent_output = tton(next_sent_output, metric_local)
        next_sent_labels = tton(next_sent_labels, metric_local)
        end_t = time.time()

        # next sentence prediction accuracy
        correct = (
            next_sent_output.argmax(axis=-1) == next_sent_labels.squeeze(1)
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
        mode="train",
        dataset_size=args.train_dataset_size,
        batch_size=args.train_global_batch_size,
        data_part_num=64,
        seq_length=args.seq_length,
        max_predictions_per_seq=args.max_predictions_per_seq,
        consistent=args.use_consistent,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode="test",
        dataset_size=1024,
        batch_size=args.val_global_batch_size,
        data_part_num=1,
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
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
    )
    print(bert_model)

    # Load the same initial parameters with lazy model.
    from utils.compare_lazy_outputs import load_params_from_lazy
    # load_params_from_lazy(
    #     bert_model.state_dict(),
    #     "/workspace/OneFlow-Benchmark/LanguageModeling/BERT/initial_model"
    # )

    assert id(bert_model.cls.predictions.decoder.weight) == id(
        bert_model.bert.embeddings.word_embeddings
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

    optimizer = build_optimizer(
        args.optim_name,
        bert_model,
        args.lr,
        args.weight_decay,
        weight_decay_excludes=["bias", "LayerNorm", "layer_norm"],
        clip_grad_max_norm=1,
        clip_grad_norm_type=2,
    )

    # steps = args.epochs * len(train_data_loader)
    steps = 1000
    warmup_steps = int(steps * args.warmup_proportion)

    lr_scheduler = PolynomialLR(optimizer, steps=steps, end_learning_rate=0.0)

    lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
        lr_scheduler, warmup_factor=0, warmup_iters=warmup_steps, warmup_method="linear"
    )

    def get_masked_lm_loss(
        logit,
        masked_lm_labels,
        label_weights,
        max_predictions_per_seq,
    ):

        label_id = flow.reshape(masked_lm_labels, [-1])

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        pre_example_loss = mlm_criterion(logit, label_id)
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
            if args.grad_acc_steps > 1:
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
                input_ids, segment_ids, input_mask, masked_lm_positions
            )

            # 2-1. loss of is_next classification result
            next_sentence_loss = self.ns_criterion(
                seq_relationship_scores.reshape(-1, 2), next_sentence_labels.reshape(-1)
            )

            masked_lm_loss = self.masked_lm_criterion(
                prediction_scores, masked_lm_ids, masked_lm_weights
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
            batch_size=args.train_global_batch_size * args.grad_acc_steps,
            # keys=["total_loss", "mlm_loss", "nsp_loss", "pred_acc"],
            keys=[]
        )

        # Train
        bert_model.train()

        for step in range(steps):
            bert_outputs, loss = pretrain(bert_graph, args.metric_local)

            if (step + 1) % args.loss_print_every_n_iters == 0:
                tton(loss) # sync

            if flow.env.get_rank() == 0:
                metric.metric_cb(step, epoch=epoch)(bert_outputs)

            # train_total_losses.append(bert_outputs["total_loss"])
    
    # with open("graph_loss.txt", 'w') as f:
    #     for loss in train_total_losses:
    #         f.write(str(loss)+'\n')

    # Eval
    # bert_model.eval()
    # val_acc = validation(
    #     epoch,
    #     len(test_data_loader),
    #     bert_eval_graph,
    #     args.val_print_every_n_iters,
    #     args.metric_local,
    # )

    # save_model(bert_model, args.checkpoint_path, epoch, val_acc, args.use_consistent)


if __name__ == "__main__":
    main()
