import argparse


def get_config():
    def str2bool(v):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser()

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
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=12, help="Number of attention heads",
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
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)
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
        "--warmup_proportion",
        type=float,
        default=0.1,
        help="Warmup propotion to total steps",
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
        "--use_fp16",
        type=str2bool,
        nargs="?",
        const=True,
        help="Whether to use use fp16",
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
