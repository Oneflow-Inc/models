import argparse
import time
import numpy as np

import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader

from utils.reporter import Reporter
from dataset.dataset import BERTDataset
from dataset.vocab import WordVocab
from model.bert import BERT
from model.language_model import BERTLM


def _parser_args():
    parser = argparse.ArgumentParser("Flags for bert training")

    parser.add_argument(
        "-c",
        "--train_dataset",
        required=False,
        type=str,
        default="data/corpus.small",
        help="train dataset for train bert",
    )
    parser.add_argument(
        "-t",
        "--test_dataset",
        type=str,
        default="data/corpus.small",
        help="test set for evaluate train set",
    )
    parser.add_argument(
        "-v",
        "--vocab_path",
        required=False,
        default="data/vocab.small",
        type=str,
        help="built vocab model path with bert-vocab",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        default="output/bert.model",
        type=str,
        help="ex)output/bert.model",
    )

    parser.add_argument(
        "-hs",
        "--hidden",
        type=int,
        default=256,
        help="hidden size of transformer model",
    )
    parser.add_argument("-l", "--layers", type=int,
                        default=8, help="number of layers")
    parser.add_argument(
        "-a", "--attn_heads", type=int, default=8, help="number of attention heads"
    )
    parser.add_argument(
        "-s", "--seq_len", type=int, default=20, help="maximum sequence len"
    )

    parser.add_argument(
        "-b", "--batch_size", type=int, default=16, help="number of batch_size"
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="number of epochs")
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

    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate of adam")
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
        "--check-dir", type=str, default="bert_check_info", help="path to image and check report save"
    )

    return parser.parse_args()


def save_model():
    pass


def train(epoch, iter_per_epoch, data_iter, graph_model, eager_model,
          criterion, eager_optim, print_interval, device):
    eager_losses = []
    graph_losses = []
    eager_times = []
    graph_times = []

    eager_correct = 0
    graph_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        # Get input data
        bert_input, segment_label, is_next, bert_label = next(data_iter)

        # Move data to specified device
        bert_input = bert_input.to(device=device)
        segment_label = segment_label.to(device=device)
        is_next = is_next.to(device=device)
        bert_label = bert_label.to(device=device)

        eager_start_time = time.time()
        # Eager forward + backward
        eager_sent_output, mask_lm_output = eager_model(
            bert_input, segment_label)

        next_loss = criterion(eager_sent_output, is_next)
        mask_loss = criterion(mask_lm_output.transpose(1, 2), bert_label)
        eager_loss = next_loss + mask_loss

        eager_loss.backward()
        eager_optim.step()
        eager_optim.zero_grad()

        # Waiting for sync
        eager_loss = eager_loss.numpy().item()
        eager_end_time = time.time()

        graph_start_time = time.time()
        # Graph forward + backward
        graph_sent_output, graph_loss = graph_model(
            bert_input, segment_label, is_next, bert_label)

        graph_loss = graph_loss.numpy().item()
        graph_end_time = time.time()

        total_element += is_next.nelement()
        # next sentence prediction accuracy
        eager_correct += (
            eager_sent_output.argmax(dim=-1).eq(is_next).sum().numpy().item()
        )
        eager_losses.append(eager_loss)

        graph_correct += (
            graph_sent_output.argmax(dim=-1).eq(is_next).sum().numpy().item()
        )
        graph_losses.append(graph_loss)

        eager_times.append(eager_end_time-eager_start_time)
        graph_times.append(graph_end_time-graph_start_time)

        if (i + 1) % print_interval == 0:
            print(
                "Epoch: {}, train iter: {}, loss(eager/graph): {:.3f}/{:.3f}, "
                "iter time(eager/graph): {:.3f}s/{:.3f}s".format(
                    epoch, (i + 1), np.mean(eager_losses), np.mean(
                        graph_losses), eager_times[-1], graph_times[-1]
                )
            )

    print(
        "Epoch {}, train iter {}, loss(eager/graph) {:.3f}/{:.3f}, "
        "total accuracy(eager/graph) {:.2f}/{:.2f}".format(
            epoch, (i + 1), np.mean(eager_losses), np.mean(graph_losses), eager_correct *
            100.0 / total_element, graph_correct * 100 / total_element
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


def validation(epoch, iter_per_epoch, data_iter, graph_model, eager_model, print_interval, device):

    eager_times = []
    graph_times = []

    eager_correct = 0
    graph_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        # Get input data
        bert_input, segment_label, is_next, bert_label = next(data_iter)
        start_t = time.time()

        bert_input = bert_input.to(device=device)
        segment_label = segment_label.to(device=device)

        eager_start_time = time.time()
        # Eager forward
        eager_sent_output, _ = eager_model(bert_input, segment_label)

        # Waiting for sync
        eager_sent_output = eager_sent_output.numpy()
        eager_end_time = time.time()

        graph_start_time = time.time()
        # Graph forward
        graph_sent_output = graph_model(
            bert_input, segment_label)

        # Waiting for sync
        graph_sent_output = graph_sent_output.numpy()
        graph_end_time = time.time()

        is_next = is_next.numpy()

        total_element += is_next.size
        # next sentence prediction accuracy
        eager_correct += (eager_sent_output.argmax(axis=-1) == is_next).sum()
        graph_correct += (graph_sent_output.argmax(axis=-1) == is_next).sum()

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
            epoch, (i+1), eager_correct *
            100.0 / total_element, graph_correct * 100 / total_element
        )
    )
    return {
        "eager_acc": eager_correct * 100.0 / total_element,
        "graph_acc": graph_correct * 100 / total_element,
        "eager_times": eager_times,
        "graph_times": graph_times,
    }


def check(args):

    if(args.with_cuda):
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(
        args.train_dataset,
        vocab,
        seq_len=args.seq_len,
        corpus_lines=args.corpus_lines,
        on_memory=args.on_memory,
    )

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = (
        BERTDataset(
            args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory
        )
        if args.test_dataset is not None
        else None
    )

    print("Creating Dataloader")
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_data_loader = (
        DataLoader(
            test_dataset, batch_size=args.batch_size, num_workers=args.num_workers
        )
        if test_dataset is not None
        else None
    )

    print("Building BERT eager model")
    eager_module = BERT(
        len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    bert_eager = BERTLM(eager_module, len(vocab))
    bert_eager.to(device)

    eager_optimizer = flow.optim.Adam(bert_eager.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.adam_weight_decay,
                                      betas=(args.adam_beta1, args.adam_beta2)
                                      )

    print("Building BERT graph model")
    graph_module = BERT(
        len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    bert_graph = BERTLM(graph_module, len(vocab))
    bert_graph.to(device)

    # Make sure graph module and eager module have the same initial parameters
    bert_graph.load_state_dict(bert_eager.state_dict())

    graph_optimizer = flow.optim.Adam(bert_graph.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.adam_weight_decay,
                                      betas=(args.adam_beta1, args.adam_beta2)
                                      )

    # TODO：add lr schedule in graph
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
            self.add_optimizer("adam", graph_optimizer)
            # self._train_data_iter = iter(train_data_loader)

        def build(self, bert_input, segment_label, is_next, bert_label):

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.bert(
                bert_input, segment_label)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            next_loss = self.nll_loss(next_sent_output, is_next)

            # 2-2. NLLLoss of predicting masked token word
            mask_loss = self.nll_loss(
                mask_lm_output.transpose(1, 2), bert_label
            )

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = next_loss + mask_loss

            loss.backward()
            return next_sent_output, loss

    bert_train_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_graph
            # self._val_data_iter = iter(val_data_loader)

        def build(self, bert_input, segment_label):

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, mask_lm_output = self.bert(
                    bert_input, segment_label)

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

        train_data_iter = iter(train_data_loader)
        train_metrics = train(epoch, len(train_data_iter),
                              train_data_iter, bert_train_graph, bert_eager, criterion,
                              eager_optimizer, args.print_interval, device)

        total_eager_losses.extend(train_metrics["eager_losses"])
        total_graph_losses.extend(train_metrics["graph_losses"])
        train_eager_acc.append(train_metrics["eager_acc"])
        train_graph_acc.append(train_metrics["graph_acc"])
        train_eager_times.extend(train_metrics["eager_times"])
        train_graph_times.extend(train_metrics["graph_times"])

        # Set eval mode
        bert_eager.eval()
        bert_graph.eval()

        test_data_iter = iter(test_data_loader)
        valid_metrics = validation(epoch, len(test_data_loader),
                                   test_data_iter, bert_eval_graph, bert_eager, args.print_interval, device)

        val_eager_acc.append(valid_metrics["eager_acc"])
        val_graph_acc.append(valid_metrics["graph_acc"])
        val_eager_times.extend(valid_metrics["eager_times"])
        val_graph_times.extend(valid_metrics["graph_times"])

        save_model()

    Reporter.save_report(
        "Bert", args.check_dir,
        total_eager_losses, total_graph_losses,
        train_eager_acc, train_graph_acc,
        val_eager_acc, val_graph_acc,
        train_eager_times, train_graph_times,
        val_eager_times, val_graph_times
    )

    Reporter.save_check_info(
        args.check_dir,
        {
            "eager_losses": total_eager_losses,
            "graph_losses": total_graph_losses,
        },
        {
            "eager_trainAcc": train_eager_acc,
            "graph_trainAcc": train_graph_acc,
        },
        {
            "eager_valAcc": val_eager_acc,
            "graph_valAcc": val_graph_acc,
        },
        {
            "eager_trainStepTime": train_eager_times[1:],
            # Remove graph compile time
            "graph_trainStepTime": train_graph_times[1:],
        },
        {
            "eager_valStepTime": val_eager_times[1:],
            "graph_valStepTime": val_graph_times[1:],
        }
    )

    Reporter.draw_check_info(args.check_dir)


if __name__ == "__main__":
    args = _parser_args()
    check(args)
