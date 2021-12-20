import os
import argparse
import shutil
import time
import yaml
import numpy as np
from numpy.lib.arraypad import pad
import torch
from torch.nn import parameter
from torch.utils.data import DataLoader
from classifier_SST2 import SST2RoBERTa
from SST2Dataset import read_data, SST2Dataset
from config import train_config
import logging

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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
    args.vocab_size = max(torch.max(input_ids), args.vocab_size)

    train_data = SST2Dataset(input_ids, attention_mask, labels)
    trainloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True)
    input_ids, attention_mask, labels = read_data('eval')

    args.vocab_size = max(torch.max(input_ids), args.vocab_size)+1
    eval_data = SST2Dataset(input_ids, attention_mask, labels)
    evalloader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=True)
    print('Data Loading Time: %.2fs' % (time.time() - start))
    time_map['t5'] = time.time()
    model = SST2RoBERTa(args, args.pretrain_dir, args.kwargs_path,
                        args.roberta_hidden_size, args.n_classes, args.is_train).to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    of_adam = torch.optim.Adam(model.parameters(), args.lr)
    print("begin to train...")
    for epoch in range(args.n_epochs):
        model.train()
        start = time.time()
        print("epoch{}:".format(epoch+1))
        i = 0
        losses = 0

        # 测试
        print(args.model_save_dir)
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        else:
            shutil.rmtree(args.model_save_dir)
            assert not os.path.exists(args.model_save_dir)
            os.mkdir(args.model_save_dir)
        

        for iter in trainloader:
            input_ids, attention_mask, labels = iter
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            i += 1
            output = model(input_ids, attention_mask)
            labels = labels.reshape(-1).to(device)
            loss = criterion(output, labels)
            losses += loss.detach()
            loss.backward()
            of_adam.step()
            of_adam.zero_grad()
            if i % 50 == 0:
                logger.info('Epoch:[{}/{}]\t loss={:.5f}'.format(epoch , args.n_epochs, loss ))
                print("batch", i, "loss", losses / (i + 1))
        print("Epoch %d training time %.2fs" % (epoch+1, time.time() - start))
        logger.info("Epoch %d training time %.2fs" % (epoch+1, time.time() - start))

        model.eval()
        g = {"correct": 0, "total": 0}
        for iter in evalloader:
            input_ids, attention_mask, labels = iter
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            labels = labels.reshape(-1).to(device)
            logits = model(input_ids, attention_mask)
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().numpy()
            get_acc(labels, logits, g)
        print(g["correct"], g["total"])
        print("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(
            epoch+1, g["correct"] * 100 / g["total"]))
        logger.info("[Epoch:{0:d} ] accuracy: {1:.1f}%".format(epoch+1, g["correct"] * 100 / g["total"]))
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        else:
            shutil.rmtree(args.model_save_dir)
            assert not os.path.exists(args.model_save_dir)
            os.mkdir(args.model_save_dir)
        torch.save(model.state_dict(), args.model_save_dir+'/RoBERT.pt')
    time_map['t6'] = time.time()
    print("Training Time: %.2fs" % (time_map['t6'] - time_map['t5']))
    logger.info("Training Time: %.2fs" % (time_map['t6'] - time_map['t5']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    with open('superparams.yaml', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)['SST2']
        for key in config.keys():
            name = '--' + key
            parser.add_argument(name, type=type(config[key]), default=config[key])
    # parser.add_argument('--lr', type=float, default=1e-6)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--model_load_dir', type=str, default='.')
    # parser.add_argument('--n_epochs', type=int, default=30)
    # parser.add_argument('--kwargs_path', type=str,
    #                     default='./pretrain_model_SST-2/parameters.json')
    # parser.add_argument('--pretrain_dir', type=str,
    #                     default='./pretrain_model_SST-2/weights')
    # parser.add_argument('--model_save_dir', type=str,
    #                     default='./pretrain_model_SST')
    # parser.add_argument('--task', type=str,
    #                     default='SST-2')
    # parser.add_argument('--vocab_size', type=int, default=30522)
    # parser.add_argument('--type_vocab_size', type=int, default=2)
    # parser.add_argument('--max_position_embeddings', type=int, default=512)
    # parser.add_argument('--hidden_size', type=int, default=768)
    # parser.add_argument('--intermediate_size', type=int, default=3072)
    # parser.add_argument('--chunk_size_feed_forward', type=int, default=0)
    # parser.add_argument('--num_layers', type=int, default=12)
    # parser.add_argument('')

    args = parser.parse_args()
    train_config(args)
    args.is_train = False
    train(args)
