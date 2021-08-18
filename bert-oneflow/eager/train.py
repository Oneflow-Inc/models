import argparse

from model.bert import BERT
from utils.ofrecord_data_utils import OfRecordDataLoader
from utils.trainer import BERTTrainer


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
        "-o",
        "--output-path",
        required=False,
        default="checkpoints",
        type=str,
        help="checkpoint path for bert model",
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

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam first beta value"
    )

    args = parser.parse_args()

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

    print("Building BERT model")
    bert = BERT(
        args.vocab_size,
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads,
    )

    print("Creating BERT Trainer")
    trainer = BERTTrainer(
        bert,
        args.vocab_size,
        train_dataloader=train_data_loader,
        test_dataloader=test_data_loader,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda,
        cuda_devices=args.cuda_devices,
        log_freq=10,
    )

    print("Trainer build finished!")
    print("Training Start......")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        print("Saving model...")
        trainer.save(epoch, args.output_path)
        if test_data_loader is not None:
            print("Running testing...")
            trainer.test(epoch)


main()
