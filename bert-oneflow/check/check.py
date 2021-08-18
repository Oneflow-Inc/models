import argparse
import time

import numpy as np
import oneflow as flow
from model.bert import BERT
from model.language_model import BERTLM
from oneflow import nn
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.reporter import Reporter


def _parser_args():
    parser = argparse.ArgumentParser("Flags for bert training")

    parser.add_argument(
        "--ofrecord-path", type=str, default="ofrecord", help="Path to ofrecord dataset"
    )
    parser.add_argument(
        "--train-batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=16, help="Validation batch size"
    )

    parser.add_argument(
        "-hs",
        "--hidden",
        type=int,
        default=256,
        help="hidden size of transformer model",
    )
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument(
        "-a", "--attn_heads", type=int, default=8, help="number of attention heads"
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, default=128, help="maximum sequence len"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="number of batch_size"
    )
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument(
        "-w", "--num_workers", type=int, default=0, help="dataloader worker size"
    )

    parser.add_argument(
        "--with_cuda",
        type=bool,
        default=True,
        help="training with CUDA: true, or false",
    )
    parser.add_argument(
        "--corpus_lines", type=int, default=None, help="total number of lines in corpus"
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )
    parser.add_argument(
        "--on_memory", type=bool, default=True, help="Loading on memory: true or false"
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument(
        "--adam-weight-decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam-beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam-beta2", type=float, default=0.999, help="adam first beta value"
    )
    parser.add_argument(
        "--print-interval", type=int, default=10, help="interval of printing"
    )
    parser.add_argument(
        "--check-dir",
        type=str,
        default="bert_check_info",
        help="path to image and check report save",
    )

    return parser.parse_args()


def train(
    epoch,
    iter_per_epoch,
    data_loader,
    graph_model,
    eager_model,
    criterion,
    eager_optim,
    eager_lr_sched,
    print_interval,
    device,
):
    eager_losses = []
    graph_losses = []
    eager_times = []
    graph_times = []

    eager_correct = 0
    graph_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        # Get input data
        (
            input_ids,
            next_sent_labels,
            input_masks,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
            masked_lm_weights,
        ) = data_loader()
        # Move data to specified device
        input_ids = input_ids.to(device=device)
        input_masks = input_masks.to(device=device)
        segment_ids = segment_ids.to(device=device)
        next_sent_labels = next_sent_labels.to(device=device)
        masked_lm_ids = masked_lm_ids.to(device=device)
        masked_lm_positions = masked_lm_positions.to(device=device)

        eager_start_time = time.time()
        # Eager forward + backward
        eager_sent_output, mask_lm_output = eager_model(
            input_ids, input_masks, segment_ids
        )

        ns_loss = criterion(eager_sent_output, next_sent_labels.squeeze(1))

        mask_lm_output = flow.gather(
            mask_lm_output,
            index=masked_lm_positions.unsqueeze(2).repeat(1, 1, args.vocab_size),
            dim=1,
        )
        mask_lm_output = flow.reshape(mask_lm_output, [-1, args.vocab_size])

        label_id_blob = flow.reshape(masked_lm_ids, [-1])

        # 2-2. NLLLoss of predicting masked token word
        lm_loss = criterion(mask_lm_output, label_id_blob)
        eager_loss = ns_loss + lm_loss

        eager_loss.backward()
        eager_optim.step()
        eager_optim.zero_grad()
        eager_lr_sched.step()

        # Waiting for sync
        eager_loss = eager_loss.numpy().item()
        eager_end_time = time.time()

        graph_start_time = time.time()
        # Graph forward + backward
        graph_sent_output, graph_loss = graph_model(
            input_ids,
            next_sent_labels,
            input_masks,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
        )

        graph_loss = graph_loss.numpy().item()
        graph_end_time = time.time()

        total_element += next_sent_labels.nelement()
        # next sentence prediction accuracy
        eager_correct += (
            eager_sent_output.argmax(dim=-1)
            .eq(next_sent_labels.squeeze(1))
            .sum()
            .numpy()
            .item()
        )
        eager_losses.append(eager_loss)

        graph_correct += (
            graph_sent_output.argmax(dim=-1)
            .eq(next_sent_labels.squeeze(1))
            .sum()
            .numpy()
            .item()
        )
        graph_losses.append(graph_loss)

        eager_times.append(eager_end_time - eager_start_time)
        graph_times.append(graph_end_time - graph_start_time)

        if (i + 1) % print_interval == 0:
            print(
                "Epoch: {}, train iter: {}, loss(eager/graph): {:.3f}/{:.3f}, "
                "iter time(eager/graph): {:.3f}s/{:.3f}s".format(
                    epoch,
                    (i + 1),
                    np.mean(eager_losses),
                    np.mean(graph_losses),
                    eager_times[-1],
                    graph_times[-1],
                )
            )

    print(
        "Epoch {}, train iter {}, loss(eager/graph) {:.3f}/{:.3f}, "
        "total accuracy(eager/graph) {:.2f}/{:.2f}".format(
            epoch,
            (i + 1),
            np.mean(eager_losses),
            np.mean(graph_losses),
            eager_correct * 100.0 / total_element,
            graph_correct * 100 / total_element,
        )
    )
    return {
        "eager_losses": eager_losses,
        "graph_losses": graph_losses,
        "eager_acc": eager_correct * 100.0 / total_element,
        "graph_acc": graph_correct * 100 / total_element,
        "eager_times": eager_times,
        "graph_times": graph_times,
    }


def validation(
    epoch, iter_per_epoch, data_loader, graph_model, eager_model, print_interval, device
):

    eager_times = []
    graph_times = []

    eager_correct = 0
    graph_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        # Get input data
        (
            input_ids,
            next_sent_labels,
            input_masks,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
            masked_lm_weights,
        ) = data_loader()

        input_ids = input_ids.to(device=device)
        input_masks = input_masks.to(device=device)
        segment_ids = segment_ids.to(device=device)

        eager_start_time = time.time()
        # Eager forward
        eager_sent_output, _ = eager_model(input_ids, input_masks, segment_ids)

        # Waiting for sync
        eager_sent_output = eager_sent_output.numpy()
        eager_end_time = time.time()

        graph_start_time = time.time()
        # Graph forward
        graph_sent_output = graph_model(input_ids, input_masks, segment_ids)

        # Waiting for sync
        graph_sent_output = graph_sent_output.numpy()
        graph_end_time = time.time()

        total_element += next_sent_labels.nelement()
        next_sent_labels = next_sent_labels.numpy()
        # next sentence prediction accuracy
        eager_correct += (
            eager_sent_output.argmax(axis=-1) == next_sent_labels.squeeze(1)
        ).sum()
        graph_correct += (
            graph_sent_output.argmax(axis=-1) == next_sent_labels.squeeze(1)
        ).sum()

        eager_times.append(eager_end_time - eager_start_time)
        graph_times.append(graph_end_time - graph_start_time)

        if (i + 1) % print_interval == 0:
            print(
                "Epoch: {}, val iter: {}, val time(eager/graph): {:.3f}s/{:.3f}s".format(
                    epoch, (i + 1), eager_times[-1], graph_times[-1]
                )
            )

    print(
        "Epoch: {}, val iter: {}, total accuracy(eager/graph) {:.2f}/{:.2f}".format(
            epoch,
            (i + 1),
            eager_correct * 100.0 / total_element,
            graph_correct * 100 / total_element,
        )
    )
    return {
        "eager_acc": eager_correct * 100 / total_element,
        "graph_acc": graph_correct * 100 / total_element,
        "eager_times": eager_times,
        "graph_times": graph_times,
    }


def check(args):

    if args.with_cuda:
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode='train',
        dataset_size=1024,
        batch_size=args.train_batch_size,
        data_part_num=1,
        seq_length=args.seq_len,
        max_predictions_per_seq=20,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode='test',
        dataset_size=1024,
        batch_size=args.val_batch_size,
        data_part_num=1,
        seq_length=args.seq_len,
        max_predictions_per_seq=20,
    )

    print("Building BERT eager model")
    eager_module = BERT(
        args.vocab_size,
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
    )
    bert_eager = BERTLM(eager_module, args.vocab_size)
    bert_eager.to(device)

    eager_optimizer = flow.optim.Adam(
        bert_eager.parameters(),
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )
    steps = args.epochs * len(train_data_loader)
    eager_cos_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        eager_optimizer, steps=steps
    )

    print("Building BERT graph model")
    graph_module = BERT(
        args.vocab_size,
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
    )
    bert_graph = BERTLM(graph_module, args.vocab_size)
    bert_graph.to(device)

    # Make sure graph module and eager module have the same initial parameters
    bert_graph.load_state_dict(bert_eager.state_dict())

    graph_optimizer = flow.optim.Adam(
        bert_graph.parameters(),
        lr=args.lr,
        weight_decay=args.adam_weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    graph_cos_lr = flow.optim.lr_scheduler.CosineAnnealingLR(
        graph_optimizer, steps=steps
    )
    # optim_schedule = ScheduledOptim(
    #         self.optim, self.bert.hidden, n_warmup_steps=warmup_steps
    #     )

    # of_nll_loss = nn.NLLLoss(ignore_index=0)
    criterion = nn.NLLLoss()
    criterion.to(device)

    class BertGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_graph
            self.nll_loss = criterion
            self.add_optimizer(graph_optimizer, lr_sch=graph_cos_lr)

        def build(
            self,
            input_ids,
            next_sent_labels,
            input_masks,
            segment_ids,
            masked_lm_ids,
            masked_lm_positions,
        ):

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.bert(
                input_ids, input_masks, segment_ids
            )

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            ns_loss = self.nll_loss(next_sent_output, next_sent_labels.squeeze(1))

            mask_lm_output = flow.gather(
                mask_lm_output,
                index=masked_lm_positions.unsqueeze(2).repeat(1, 1, args.vocab_size),
                dim=1,
            )
            mask_lm_output = flow.reshape(mask_lm_output, [-1, args.vocab_size])

            label_id_blob = flow.reshape(masked_lm_ids, [-1])

            # 2-2. NLLLoss of predicting masked token word
            lm_loss = self.nll_loss(mask_lm_output, label_id_blob)

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = ns_loss + lm_loss

            loss.backward()
            return next_sent_output, loss

    bert_train_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_graph

        def build(self, input_ids, input_masks, segment_ids):

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, _ = self.bert(input_ids, input_masks, segment_ids)

            return next_sent_output

    bert_eval_graph = BertEvalGraph()

    total_eager_losses = []
    total_graph_losses = []
    train_eager_acc = []
    train_graph_acc = []
    train_eager_times = []
    train_graph_times = []
    val_eager_acc = []
    val_graph_acc = []
    val_eager_times = []
    val_graph_times = []

    for epoch in range(args.epochs):
        # Set train mode
        bert_eager.train()
        bert_graph.train()

        train_metrics = train(
            epoch,
            len(train_data_loader),
            train_data_loader,
            bert_train_graph,
            bert_eager,
            criterion,
            eager_optimizer,
            eager_cos_lr,
            args.print_interval,
            device,
        )

        total_eager_losses.extend(train_metrics["eager_losses"])
        total_graph_losses.extend(train_metrics["graph_losses"])
        train_eager_acc.append(train_metrics["eager_acc"])
        train_graph_acc.append(train_metrics["graph_acc"])
        train_eager_times.extend(train_metrics["eager_times"])
        train_graph_times.extend(train_metrics["graph_times"])

        # Set eval mode
        bert_eager.eval()
        bert_graph.eval()

        valid_metrics = validation(
            epoch,
            len(test_data_loader),
            test_data_loader,
            bert_eval_graph,
            bert_eager,
            args.print_interval,
            device,
        )

        val_eager_acc.append(valid_metrics["eager_acc"])
        val_graph_acc.append(valid_metrics["graph_acc"])
        val_eager_times.extend(valid_metrics["eager_times"])
        val_graph_times.extend(valid_metrics["graph_times"])

    Reporter.save_report(
        "Bert",
        args.check_dir,
        total_eager_losses,
        total_graph_losses,
        train_eager_acc,
        train_graph_acc,
        val_eager_acc,
        val_graph_acc,
        train_eager_times,
        train_graph_times,
        val_eager_times,
        val_graph_times,
    )

    Reporter.save_check_info(
        args.check_dir,
        {"eager_losses": total_eager_losses, "graph_losses": total_graph_losses,},
        {"eager_trainAcc": train_eager_acc, "graph_trainAcc": train_graph_acc,},
        {"eager_valAcc": val_eager_acc, "graph_valAcc": val_graph_acc,},
        {
            "eager_trainStepTime": train_eager_times[1:],
            # Remove graph compile time
            "graph_trainStepTime": train_graph_times[1:],
        },
        {
            "eager_valStepTime": val_eager_times[1:],
            "graph_valStepTime": val_graph_times[1:],
        },
    )

    Reporter.draw_check_info(args.check_dir)


if __name__ == "__main__":
    args = _parser_args()
    check(args)
