# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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

from abc import ABC
from abc import abstractmethod
import math
from .gpt2_tokenization import GPT2Tokenizer


def build_tokenizer(vocab_file, merges_file, tokenizer_type="GPT2BPETokenizer"):
    """Select and instantiate the tokenizer."""
    if tokenizer_type == "GPT2BPETokenizer":
        tokenizer = _GPT2BPETokenizer(vocab_file, merges_file)
    else:
        raise NotImplementedError(
            "{} tokenizer is not implemented.".format(tokenizer_type)
        )
    return tokenizer


def pad_vocab_size(vocab_size, alignment, model_parallel_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    assert isinstance(alignment, int)
    if alignment == 0:
        return vocab_size

    alignment *= model_parallel_size

    padded_vocab_size = int(math.ceil(vocab_size / alignment)) * alignment
    print(
        " > padded vocab (size: {}) with {} dummy tokens (new size: {})".format(
            vocab_size, padded_vocab_size - vocab_size, padded_vocab_size
        )
    )
    return padded_vocab_size


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} tokenizer".format(self.name)
        )


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = "GPT2 BPE"
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(
            vocab_file, merge_file, errors="replace", special_tokens=[], max_len=None
        )
        self.eod_id = self.tokenizer.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id
