import argparse


def get_args(print_args=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dnn_use_bn", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--cross_num", type=int, default=3)
    parser.add_argument("--cross_parameterization", type=str, default="vector")
    parser.add_argument("--dnn_activation", type=str, default="relu")
    parser.add_argument("--dnn_hidden_units", type=str, default="400,400,400")
    parser.add_argument("--embedding_dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--loss", type=str, default="binary_crossentropy")
    parser.add_argument("--metrics", type=str, default="binary_crossentropy,auc")
    parser.add_argument("--dnn_dropout", type=float, default=0.2)
    parser.add_argument("--l2_reg_embedding", type=int, default=0.005)
    parser.add_argument("--l2_reg_cross", type=float, default=0.00001)
    parser.add_argument("--l2_reg_dnn", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--task", type=str, default="binary")
    parser.add_argument("--model_path", type=str, default="./log")
    parser.add_argument("--model_dir", type=str, default="./log/models/frappe")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


if __name__ == "__main__":

    args = get_args()


