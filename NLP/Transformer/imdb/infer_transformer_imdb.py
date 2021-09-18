import numpy as np
import os
import argparse
import json

import oneflow as flow

from utils import pad_sequences
from model import TransformerEncoderModel

parser = argparse.ArgumentParser()

parser.add_argument("--sequence_len", type=int, default=128)  # src_len
parser.add_argument("--vocab_sz", type=int, default=10000)  # d_model
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_encoder_layers", type=int, default=6)
parser.add_argument("--n_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=1024)

parser.add_argument("--imdb_path", type=str, default="../../imdb")
parser.add_argument("--load_dir", type=str, default=".")

parser.add_argument("--text", type=str, default="This film is too bad.")

args = parser.parse_args()
args.n_classes = 2  # tgt_len


def predict(model, text):
    model.eval()
    logits = model(flow.tensor(text))
    logits = flow.softmax(logits)
    label = flow.argmax(logits)

    return label.numpy(), logits.numpy()


def inference(text):

    with open(os.path.join(args.imdb_path, "word_index.json")) as f:
        word_index = json.load(f)
    word_index = {k: (v + 2) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2

    model = TransformerEncoderModel(
        emb_sz=args.vocab_sz,
        n_classes=args.n_classes,
        d_model=args.d_model,
        nhead=args.n_head,
        num_encoder_layers=args.n_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_first=True,
    )
    model.load_state_dict(flow.load(args.load_dir))

    import re

    text = re.sub("[^a-zA-Z']", " ", text)
    text = list(map(lambda x: x.lower(), text.split()))
    text.insert(0, "<START>")
    text = [
        list(
            map(
                lambda x: word_index[x] if x in word_index else word_index["<UNK>"],
                text,
            )
        )
    ]
    text = pad_sequences(
        text, value=word_index["<PAD>"], padding="post", maxlen=args.sequence_len
    )
    text = np.array(text, dtype=np.int32)
    label, logits = predict(model, text)
    print(label, logits)


if __name__ == "__main__":

    inference(args.text)
