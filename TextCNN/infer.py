import oneflow as flow

import argparse
import numpy as np
import pickle
import json

from model import textCNN
import utils


def _parse_args():
    parser = argparse.ArgumentParser("flags for test textcnn")
    parser.add_argument(
        "--model_path", type=str, default="./checkpoints", help="model path"
    )
    parser.add_argument(
        "--vocab_path", type=str, default="./vocab.pkl", help="vocab path"
    )
    parser.add_argument(
        "--config_path", type=str, default="./config.json", help="config path"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--text", type=str, default="", help="input text")
    return parser.parse_args()


def main(args):

    device = flow.device("cpu") if args.no_cuda else flow.device("cuda")
    with open(args.config_path, "r") as f:
        config = json.load(f)
    with open(args.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    textcnn = textCNN(
        word_emb_dim=config["word_emb_dim"],
        vocab_size=len(vocab),
        dim_channel=config["dim_channel"],
        kernel_wins=config["kernel_wins"],
        dropout_rate=config["dropout_rate"],
        num_class=config["num_class"],
        max_seq_len=config["max_seq_len"],
    )
    textcnn.load_state_dict(flow.load(args.model_path))
    textcnn.eval()
    textcnn.to(device)
    text = utils.clean_str(args.text)
    text = [utils.tokenizer(text)]
    input = flow.tensor(
        np.array(utils.tensorize_data(text, vocab, max_len=200)), dtype=flow.long
    ).to(device)
    predictions = textcnn(input).softmax()
    predictions = predictions.numpy()
    clsidx = np.argmax(predictions)
    print("predict prob: %f, class name: %s" % (np.max(predictions), clsidx))


if __name__ == "__main__":
    args = _parse_args()
    main(args)
