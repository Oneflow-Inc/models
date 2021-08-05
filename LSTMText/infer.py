import argparse
import json
import os

import numpy as np
import oneflow as flow

from model import LSTMText
from utils import pad_sequences

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--emb_dim', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--nfc', type=int, default=128)
parser.add_argument('--sequence_length', type=int, default=128)
parser.add_argument('--model_load_dir', type=str, default='pretrain_model')
parser.add_argument('--imdb_path', type=str, default='../imdb')
parser.add_argument('--text', type=str, default='This film is too bad.')

args = parser.parse_args()
args.emb_num = 50000
args.n_classes = 2


def predict(model, text):
    model.eval()
    text = flow.tensor(text).to('cuda')
    text.unsqueeze(0)
    logits = model(text)
    logits = flow.softmax(logits)
    label = flow.argmax(logits)
    
    return label.numpy(), logits.numpy()


def inference(text):
    with open(os.path.join(args.imdb_path, 'word_index.json')) as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    
    model = LSTMText(args.emb_num, args.emb_dim, hidden_size=args.hidden_size,
                     nfc=args.nfc, n_classes=args.n_classes, batch_size=args.batch_size)
    model.load_state_dict(flow.load(args.model_load_dir))
    model.to('cuda')
    import re
    text = re.sub("[^a-zA-Z']", " ", text)
    text = list(map(lambda x: x.lower(), text.split()))
    text.insert(0, "<START>")
    text = [list(map(lambda x: word_index[x]
    if x in word_index else word_index["<UNK>"], text))]
    text = pad_sequences(
        text, value=word_index["<PAD>"], padding='post', maxlen=args.sequence_length)
    text = np.array(text, dtype=np.int32)
    label, logits = predict(model, text)
    print(label, logits)


if __name__ == '__main__':
    inference(args.text)
