import argparse
import time

import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader

from dataset.dataset import BERTDataset
from dataset.vocab import WordVocab
from model.bert import BERT
from model.language_model import BERTLM


def train(epoch, iter_per_epoch, data_iter, graph, print_interval):
    avg_loss = 0.0
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        bert_input, segment_label, is_next, bert_label = next(data_iter)
        start_t = time.time()

        next_sent_output, is_next, loss = graph(
            bert_input, segment_label, is_next, bert_label)

        loss = loss.numpy().item()
        end_t = time.time()

        # flow.save(self.bert.state_dict(), "checkpoints/bert_%d_loss_%f" % (i, loss.numpy().item()))

        # next sentence prediction accuracy
        correct = (
            next_sent_output.argmax(dim=-1).eq(is_next).sum().numpy().item()
        )
        avg_loss += loss
        total_correct += correct
        total_element += is_next.nelement()

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, train iter {}, loss {}, train time: {}".format(
                    epoch, (i + 1), avg_loss / (i + 1), end_t - start_t
                )
            )

    print("total_correct >>>>>>>>>>>>>> ", total_correct)
    print("total_element >>>>>>>>>>>>>> ", total_element)
    print(
        "Epoch {}, train iter {}, loss {}, total accuracy {}".format(
            epoch, (i+1), avg_loss / (i + 1), total_correct *
            100.0 / total_element
        )
    )


def validation(epoch, iter_per_epoch, data_iter, graph, print_interval):
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        bert_input, segment_label, is_next, bert_label = next(data_iter)
        start_t = time.time()

        next_sent_output, is_next = graph(
            bert_input, segment_label, is_next, bert_label)

        next_sent_output = next_sent_output.numpy()
        is_next = is_next.numpy()
        end_t = time.time()

        # flow.save(self.bert.state_dict(), "checkpoints/bert_%d_loss_%f" % (i, loss.numpy().item()))

        # next sentence prediction accuracy
        correct = (next_sent_output.argmax(axis=-1) == is_next).sum()
        total_correct += correct
        total_element += is_next.size

        if (i + 1) % print_interval == 0:
            print(
                "Epoch {}, val iter {}, val time: {}".format(
                    epoch, (i + 1), end_t - start_t
                )
            )

    print("total_correct >>>>>>>>>>>>>> ", total_correct)
    print("total_element >>>>>>>>>>>>>> ", total_element)
    print(
        "Epoch {}, val iter {}, total accuracy {}".format(
            epoch, (i+1), total_correct *
            100.0 / total_element
        )
    )


def main():

    parser = argparse.ArgumentParser()

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

    parser.add_argument("--lr", type=float, default=1e-3,
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

    args = parser.parse_args()

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

    print("Building BERT model")
    bert_module = BERT(
        len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    bert_module.to(device)

    bert_model = BERTLM(bert_module, len(vocab))
    bert_model.to(device)

    optimizer = flow.optim.Adam(bert_model.parameters(),
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
            self.bert = bert_model
            self.nll_loss = criterion
            self.add_optimizer("adam", optimizer)
            # self._train_data_iter = iter(train_data_loader)

        def build(self, bert_input, segment_label, is_next, bert_label):
            # try:
            #     bert_input, segment_label, is_next, bert_label = next(self._train_data_iter)
            # except StopIteration:
            #     self._train_data_iter = iter(train_data_loader)
            #     bert_input, segment_label, is_next, bert_label = next(self._train_data_iter)
            bert_input = bert_input.to(device=device)
            segment_label = segment_label.to(device=device)
            is_next = is_next.to(device=device)
            bert_label = bert_label.to(device=device)

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
            return next_sent_output, is_next, loss

    bert_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            # self._val_data_iter = iter(val_data_loader)

        def build(self, bert_input, segment_label, is_next, bert_label):

            bert_input = bert_input.to(device=device)
            segment_label = segment_label.to(device=device)
            is_next = is_next.to(device=device)
            bert_label = bert_label.to(device=device)

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, mask_lm_output = self.bert(
                    bert_input, segment_label)

            return next_sent_output, is_next

    bert_eval_graph = BertEvalGraph()

    for epoch in range(args.epochs):
        # Train
        bert_model.train()

        train_data_iter = iter(train_data_loader)
        train(epoch, len(train_data_iter),
              train_data_iter, bert_graph, args.print_interval)

        # Eval
        bert_model.eval()

        test_data_iter = iter(test_data_loader)
        validation(epoch, len(test_data_loader),
                   test_data_iter, bert_eval_graph, args.print_interval)

        # trainer.train(epoch)
        # print("Saving model...")
        # trainer.save(epoch, args.output_path)
        # if test_data_loader is not None:
        #     print("Running testing...")
        #     trainer.test(epoch)
if __name__ == "__main__":
    main()
