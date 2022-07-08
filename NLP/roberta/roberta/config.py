import argparse

def infer_config(args):
    if args.task == 'SST-2':
        args.n_classes = 2
        args.roberta_hidden_size = 768
        args.is_train = False
    elif args.task == 'MNLI':
        args.n_classes = 3
        args.roberta_hidden_size = 768
        args.is_train = False

def train_config(args):
    if args.task == 'SST-2':
        args.n_classes = 2
        args.roberta_hidden_size = 768
        args.is_train = True
    elif args.task == 'MNLI':
        args.n_classes = 3
        args.roberta_hidden_size = 768
        args.is_train = True