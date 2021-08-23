# https://github.com/Kenneth111/TransformerDemo/blob/master/predict_odd_numbers.py
import sys
import argparse

import oneflow as flow

sys.path.append("../")
from model import TransformerModel

TO_CUDA = True

parser = argparse.ArgumentParser()

parser.add_argument("--vocab_sz", type=int, default=50000)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_encoder_layers", type=int, default=6)
parser.add_argument("--n_decoder_layers", type=int, default=6)
parser.add_argument("--dim_feedforward", type=int, default=1024)

parser.add_argument("--load_dir", type=str, default=".")
parser.add_argument("--input_start", type=int)

args = parser.parse_args()


def to_cuda(tensor, flag=TO_CUDA, where="cuda"):
    if flag:
        return tensor.to(where)
    else:
        return tensor


MAX_LEN = 3


def main():

    voc_size = args.vocab_sz
    print("Setting model...", end="")
    model = TransformerModel(
        input_sz=voc_size,
        output_sz=voc_size,
        d_model=args.d_model,
        nhead=args.n_head,
        num_encoder_layers=args.n_encoder_layers,
        num_decoder_layers=args.n_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model.load_state_dict(flow.load(args.load_dir))
    model = to_cuda(model)
    print("Done")

    print("Inference:")
    num = args.input_start
    if num % 2 != 0:
        print("The input number must be an even number.")
        return
    if num > args.vocab_sz - MAX_LEN * 2:
        print("The input sequence may be out of range.")
        return

    input_nums = [num + i * 2 for i in range(MAX_LEN)]
    src = to_cuda(flow.tensor(input_nums)).unsqueeze(1)
    pred = [0]
    for i in range(MAX_LEN):
        inp = to_cuda(flow.tensor(pred)).unsqueeze(1)
        output = model(src, inp)
        out_num = output.argmax(2)[-1].numpy()[0]
        pred.append(out_num)
    print("input:", input_nums)
    print("pred:", pred)


if __name__ == "__main__":

    main()
