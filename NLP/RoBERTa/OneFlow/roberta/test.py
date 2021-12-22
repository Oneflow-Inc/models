import os
import argparse
import shutil
import time
import yaml
import numpy as np
import oneflow
from oneflow.utils.data import DataLoader
from classifier_SST2 import SST2RoBERTa
from SST2Dataset import read_data, SST2Dataset
from config import train_config
import logging

use_cuda = True
if oneflow.cuda.is_available() and use_cuda:
    device = oneflow.device("cuda")
else:
    device = oneflow.device("cpu")

seed = 23
# 设置随机种子
def set_rng_seed(seed):
    np.random.seed(seed)
    oneflow.manual_seed(seed)
    oneflow.cuda.manual_seed(seed)
    oneflow.cuda.manual_seed_all(seed)
    # 反卷机的方法将是固定的
    oneflow.backends.cudnn.deterministic = True


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

logger = get_logger('exp.log')

time_map = {}
def get_acc(labels, logits, g):
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g["total"] += labels.shape[0]
    g["correct"] += right_count


def train(args):

    time_map['t1'] = time.time()
    print("begin to load data...")
    start = time.time()
    input_ids, attention_mask, labels = read_data('train')
    train_data = SST2Dataset(input_ids, attention_mask, labels)
    trainloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False)
    input_ids, attention_mask, labels = read_data('eval')
    eval_data = SST2Dataset(input_ids, attention_mask, labels)
    evalloader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=False)
    print('Data Loading Time: %.2fs' % (time.time() - start))
    time_map['t5'] = time.time()
    model = SST2RoBERTa(args, args.pretrain_dir, args.kwargs_path,
                        args.roberta_hidden_size, args.n_classes, args.is_train).to(device)
    criterion = oneflow.nn.CrossEntropyLoss().to(device)
    of_adam = oneflow.optim.Adam(model.parameters(), args.lr)
    print("begin to train...")

    with oneflow.no_grad():
        model.eval()
        losses = 0
        for iter in trainloader:
            input_ids, attention_mask, labels = iter
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            output = model(input_ids, attention_mask)
            labels = labels.reshape(-1).to(device)
            loss = criterion(output, labels)
            losses += loss.detach()
        print(losses)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    with open('superparams.yaml', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)['SST2']
        for key in config.keys():
            name = '--' + key
            parser.add_argument(name, type=type(config[key]), default=config[key])
    args = parser.parse_args()
    train_config(args)
    args.is_train = False
    train(args)
