
import os
import sys
import random
import argparse
import numpy as np
import oneflow as flow
from model import GPT2LMHeadModel
from utils import convert_pt_checkpoint_to_of
from model_config import GPT2Config
from sample import sample_sequence
from tokenizer.tokenizer import build_tokenizer

from transformers import GPT2Tokenizer, GPT2Model

def text_generator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    flow.manual_seed(args.seed)

    device = flow.device("cuda")

    tokenizer = build_tokenizer(vocab_file='vocab.json', merges_file='merge.txt')
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    # convert_pt_checkpoint_to_of(model, pt_checkpoint_path="gpt2-pytorch_model.bin", of_checkpoint_path="gpt2_oneflow_model")

    state_dict = flow.load("gpt2_oneflow_model")
    model.load_state_dict(state_dict)
    model.tie_embeddings()

    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = config.n_ctx // 2
    elif args.length > config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % config.n_ctx)
    
    text = args.text
    print(text)

    context_tokens = tokenizer.tokenize(text)
    generated = 0
    for _ in range(args.nsamples // args.batch_size):
        out = sample_sequence(
            model=model, length=args.length,
            context=context_tokens if not args.unconditional else None,
            start_token=tokenizer.vocab['<|endoftext|>'] if args.unconditional else None,
            batch_size=args.batch_size,
            temperature=args.temperature, top_k=args.top_k, device=device
        )
        out = out[:, len(context_tokens):].tolist()
        for i in range(args.batch_size):
            generated += 1
            text = tokenizer.detokenize(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)

if __name__ == '__main__':
    text_generator()