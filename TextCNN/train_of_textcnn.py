import oneflow.experimental as flow

import os
import numpy as np
import argparse
import shutil
import pickle
from tqdm import tqdm
import json

from model import textCNN
from training import train,_eval
import utils

flow.enable_eager_execution()

def _parse_args():
    parser = argparse.ArgumentParser("flags for train TextCNN")
    parser.add_argument(
        "--save_checkpoint_path", type=str, default="./checkpoints", help="save checkpoint root dir"
    )
    parser.add_argument(
        "--save_vocab_path", type=str, default="vocab.pkl", help="save vocab root dir"
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="", help="load checkpoint"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./aclImdb", help="dataset path"
    )
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # training hyper-parameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="training epochs"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="train batch size"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=16, help="val batch size"
    )
    parser.add_argument(
        "--word_emb_dim", type=int, default=100, help="dimensions of word embeddings"
    )
    parser.add_argument(
        "--conv_channel_size", type=int, default=64, help="channel size of Conv2d"
    )
    parser.add_argument(
        "--kernel_size", nargs='+', type=int, default=[3,4,5], help="channel size of Conv2d"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.5, help="dropout rate"
    )
    parser.add_argument(
        "--num_class", type=int, default=2, help="number of classes"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=200, help="maximum allowed sequence length"
    )
    return parser.parse_args()


def batch_loader(data, 
                label, 
                batch_size,
                shuffle = True):
    if shuffle:
        permu = np.random.permutation(len(data))
        data, label = data[permu], label[permu]
    batch_n = len(data) // batch_size
    x_batch = [flow.tensor(data[i * batch_size:(i * batch_size + batch_size)],dtype = flow.long) for i in range(batch_n)]
    y_batch = [flow.tensor(label[i * batch_size:(i * batch_size + batch_size)],dtype = flow.long)for i in range(batch_n)]
    if batch_size*batch_n < len(data):
        x_batch += [flow.tensor(data[batch_size*batch_n:len(label)],dtype = flow.long)]
        y_batch += [flow.tensor(label[batch_size*batch_n:len(label)],dtype = flow.long)]
    return x_batch, y_batch


def main(args):
    config_dct = { 'word_emb_dim':args.word_emb_dim,
                    'dim_channel':args.conv_channel_size, 
                    'kernel_wins':args.kernel_size, 
                    'dropout_rate':args.dropout_rate, 
                    'num_class':args.num_class,
                    'max_seq_len':args.max_seq_len
    }
    with open('config.json', 'w') as f:
        json.dump(config_dct, f)

    device = flow.device('cpu') if args.no_cuda else flow.device('cuda')
    
    x_train,y_train = utils.load_dataset(os.path.join(args.dataset_path,'train'))
    x_test,y_test = utils.load_dataset(os.path.join(args.dataset_path,'test'))
    vocab_dct = utils.build_vocab(x_train+x_test)

    with open(args.save_vocab_path, 'wb') as f:
        pickle.dump(vocab_dct, f)

    x_train = utils.tensorize_data(x_train,vocab_dct)
    x_test = utils.tensorize_data(x_test,vocab_dct)

    y_train, x_train= np.array(y_train),np.array(x_train)
    y_test, x_test = np.array(y_test), np.array(x_test)

    textcnn = textCNN(word_emb_dim = args.word_emb_dim,
                     vocab_size = len(vocab_dct), 
                     dim_channel = args.conv_channel_size, 
                     kernel_wins = args.kernel_size, 
                     dropout_rate = args.dropout_rate, 
                     num_class = args.num_class,
                     max_seq_len = args.max_seq_len)
    textcnn.to(device)
    optimizer = flow.optim.Adam(textcnn.parameters(), lr = args.learning_rate)
    loss_func = flow.nn.BCEWithLogitsLoss().to(device)

    if args.load_checkpoint != "":
        textcnn.load_state_dict(flow.load(args.load_checkpoint))
    
    train(model = textcnn,
          device = device,
          train_data = (x_train,y_train),
          dev_data = (x_test, y_test),
          loss_func = loss_func,
          optimizer = optimizer,
          epochs = args.epochs,
          train_batch_size = args.train_batch_size,
          eval_batch_size = args.val_batch_size,
          save_path = args.save_checkpoint_path)


if __name__ == '__main__':
    args = _parse_args()
    main(args)