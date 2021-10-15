#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import oneflow as flow
import numpy as np

from model_config import GPT2Config

from model import GPT2LMHeadModel
from tokenizer import build_tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

def set_seed(args):
    np.random.seed(args.seed)
    flow.manual_seed(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < flow.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # todo: support top_p
    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = flow.sort(logits, descending=True)
    #     cumulative_probs = flow.cumsum(flow.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=1, top_p=0.0, device='cuda'):
    context = flow.tensor(context, dtype=flow.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    past_key_values = None
    with flow.no_grad():
        for _ in trange(length):
            outputs = model(generated, past_key_values=past_key_values, use_cache=True)
            logits, past_key_values = outputs[:2]
            next_token_logits = logits[:, -1, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = filtered_logits.softmax(-1)
            next_token = probs.argmax(-1)
            # next_token = flow.multinomial(flow.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = flow.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_file", default="gpt2-vocab.json", type=str)
    parser.add_argument("--merges_file", default="gpt2-merges.txt", type=str)
    parser.add_argument("--restore_file", default="gpt2_oneflow_model", type=str, help="Path to pre-trained model")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    args.device = flow.device("cuda" if not args.no_cuda else "cpu")

    set_seed(args)

    tokenizer = build_tokenizer(vocab_file=args.vocab_file, merges_file=args.merges_file, tokenizer_type="GPT2BPETokenizer")
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    if args.restore_file is not None:
        model.load_state_dict(flow.load(args.restore_file))
    model.lm_head.weight = model.transformer.wte.weight
    model.to(args.device)
    model.eval()

    if args.length < 0 and config.max_position_embeddings > 0:
        args.length = config.max_position_embeddings
    elif 0 < config.max_position_embeddings < args.length:
        args.length = config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    print(args)
    while True:
        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        context_tokens = tokenizer.tokenize(raw_text)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
        )
        out = out[0, len(context_tokens):].tolist()
        text = tokenizer.detokenize(out)
        print(text)
        if args.prompt:
            break
    return text


if __name__ == '__main__':
    main()
