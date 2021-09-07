import argparse

import oneflow as flow
from oneflow.utils.data import DataLoader

from model_config import GPT2Config
from model import GPT2LMHeadModel
from trainer import Trainer
from gpt_dataset import GPTDataset
from tokenizer import build_tokenizer

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", required=False, type=str, default="data/corpus.small", help="train dataset")
    parser.add_argument("--test_dataset", type=str, default="data/corpus.small", help="test set for evaluation")
    parser.add_argument("--vocab_file", required=False, default="vocab.json", type=str)
    parser.add_argument("--merges_file", required=False, default="merge.txt", type=str)
    parser.add_argument("--output_path", required=False, default="output/model", type=str, help="save path")

    parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")

    parser.add_argument("--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
    parser.add_argument("--device_id", type=int, default=0)

    args = parser.parse_args()

    device = flow.device(f"cuda:{args.device_id}" if args.device_id >= 0 else "cpu")

    print("building tokenizer")
    tokenizer = build_tokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file, tokenizer_type="GPT2BPETokenizer")

    print("building train dataset")
    train_dataset = GPTDataset(args.train_dataset, tokenizer, args.seq_len)

    print("building test dataset")
    test_dataset = GPTDataset(args.test_dataset, tokenizer, args.seq_len)
    
    print("building train dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    print("building test dataloader")
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    print("building model")
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    trainer = Trainer(
        model,
        train_dataloader=train_data_loader,
        test_dataloader=test_data_loader,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        device=device,
        output_path=args.output_path
    )

    print("begin training")
    trainer.train(args.epochs)

if __name__=='__main__':
    main()
