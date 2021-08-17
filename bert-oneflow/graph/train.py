import argparse
import time
import os

import oneflow as flow
from model.bert import BERT
from model.language_model import BERTLM
from oneflow import nn
from utils.ofrecord_data_utils import OfRecordDataLoader


def save_model(module: nn.Module, checkpoint_path: str, epoch: int, acc: float):
    flow.save(module.state_dict(), os.path.join(checkpoint_path,
                                                "epoch_%d_val_acc_%f" % (epoch, acc)))


def train(epoch, iter_per_epoch, graph, print_interval):
    total_loss = 0.0
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
            next_sent_output.argmax(
                dim=-1).eq(next_sent_labels.squeeze(1)).sum().numpy().item()
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
            epoch, (i+1), total_loss / (i + 1), total_correct *
            100.0 / total_element
        )
    )


def validation(epoch: int, iter_per_epoch: int, graph: nn.Graph, print_interval: int) -> float:
    total_correct = 0
    total_element = 0
    for i in range(iter_per_epoch):

        start_t = time.time()

        next_sent_output, next_sent_labels = graph()

        next_sent_output = next_sent_output.numpy()
        next_sent_labels = next_sent_labels.numpy()
        end_t = time.time()

        # next sentence prediction accuracy
        correct = (next_sent_output.argmax(axis=-1) ==
                   next_sent_labels.squeeze(1)).sum()
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
            epoch, (i+1), total_correct *
            100.0 / total_element
        )
    )
    return total_correct / total_element


def main():

    parser = argparse.ArgumentParser()

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
        help="Hidden size of transformer model",
    )
    parser.add_argument("-l", "--layers", type=int,
                        default=8, help="Number of layers")
    parser.add_argument(
        "-a", "--attn_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "-s", "--seq-len", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")

    parser.add_argument(
        "--with_cuda",
        type=bool,
        default=True,
        help="Training with CUDA: true, or false",
    )
    parser.add_argument(
        "--corpus_lines", type=int, default=None, help="Total number of lines in corpus"
    )
    parser.add_argument(
        "--cuda_devices", type=int, nargs="+", default=None, help="CUDA device ids"
    )

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate of adam")
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
        "--checkpoint-path", type=str, default="checkpoints", help="Path to model saving"
    )

    args = parser.parse_args()

    if(args.with_cuda):
        device = flow.device("cuda")
    else:
        device = flow.device("cpu")

    print("Device is: ", device)

    print("Creating Dataloader")
    train_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode='train', dataset_size=1024, batch_size=args.train_batch_size, data_part_num=1, seq_length=args.seq_len,
        max_predictions_per_seq=20,
    )

    test_data_loader = OfRecordDataLoader(
        ofrecord_dir=args.ofrecord_path,
        mode='test', dataset_size=1024, batch_size=args.val_batch_size, data_part_num=1, seq_length=args.seq_len,
        max_predictions_per_seq=20,
    )

    print("Building BERT model")
    bert_module = BERT(
        args.vocab_size, hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads
    )
    bert_module.to(device)

    bert_model = BERTLM(bert_module, args.vocab_size)
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
            self._train_data_loader = train_data_loader

        def build(self):

            input_ids, next_sent_labels, input_masks, segment_ids, masked_lm_ids, \
                masked_lm_positions, masked_lm_weights = self._train_data_loader()
            input_ids = input_ids.to(device=device)
            input_masks = input_masks.to(device=device)
            segment_ids = segment_ids.to(device=device)
            next_sent_labels = next_sent_labels.to(device=device)
            masked_lm_ids = masked_lm_ids.to(device=device)
            masked_lm_positions = masked_lm_positions.to(device=device)

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.bert(
                input_ids, input_masks, segment_ids)

            # 2-1. NLL(negative log likelihood) loss of is_next classification result
            ns_loss = self.nll_loss(
                next_sent_output, next_sent_labels.squeeze(1))

            mask_lm_output = flow.gather(mask_lm_output, index=masked_lm_positions.unsqueeze(
                2).repeat(1, 1, args.vocab_size), dim=1)
            mask_lm_output = flow.reshape(
                mask_lm_output, [-1, args.vocab_size])

            label_id_blob = flow.reshape(masked_lm_ids, [-1])

            # 2-2. NLLLoss of predicting masked token word
            lm_loss = self.nll_loss(mask_lm_output, label_id_blob)

            # 2-3. Adding next_loss and mask_loss : 3.4 Pre-training Procedure
            loss = ns_loss + lm_loss

            loss.backward()
            return next_sent_output, next_sent_labels, loss

    bert_graph = BertGraph()

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_model
            self._test_data_loader = test_data_loader

        def build(self):
            input_ids, next_sent_labels, input_masks, segment_ids, masked_lm_ids, \
                masked_lm_positions, masked_lm_weights = self._test_data_loader()
            input_ids = input_ids.to(device=device)
            input_masks = input_masks.to(device=device)
            segment_ids = segment_ids.to(device=device)
            next_sent_labels = next_sent_labels.to(device=device)
            masked_lm_ids = masked_lm_ids.to(device=device)
            masked_lm_positions = masked_lm_positions.to(device)

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                next_sent_output, _ = self.bert(
                    input_ids, input_masks, segment_ids)

            return next_sent_output, next_sent_labels

    bert_eval_graph = BertEvalGraph()

    for epoch in range(args.epochs):
        # Train
        bert_model.train()
        train(epoch, len(train_data_loader), bert_graph, args.print_interval)

        # Eval
        bert_model.eval()
        val_acc = validation(epoch, len(test_data_loader),
                             bert_eval_graph, args.print_interval*10)

        print("Saveing model ...")
        save_model(bert_model, args.checkpoint_path, epoch, val_acc)


if __name__ == "__main__":
    main()
