
from __future__ import absolute_import, division, print_function, unicode_literals
# file_utils.py

"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""

import json
import logging
import os
import shutil
import tempfile
from functools import wraps
from hashlib import sha256
import sys
from io import open

import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm

from urllib.parse import urlparse

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                   Path.home() / '.torch_pretrained_bert'))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                              os.path.join(os.path.expanduser("~"), '.torch_pretrained_bert'))

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError("HEAD request failed for url {} with status code {}"
                          .format(url, response.status_code))
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w', encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename):
    '''
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    '''
    collection = set()
    with open(filename, 'r', encoding='utf-8') as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection


def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext

# wordpiece.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes. Provided as is from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py"""


import collections
import logging
import os
import unicodedata
from io import open


logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': ".torch_pretrained_bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': ".torch_pretrained_bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': ".torch_pretrained_bert/bert-base-cased-vocab.txt",
    'bert-large-cased': ".torch_pretrained_bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    #.torch_pretrained_bert/bert-base-uncased-vocab.txt
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer(object):

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        #False
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize

        if do_basic_tokenize:
          self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        if self.do_basic_tokenize:
          split_tokens = []
          for token in self.basic_tokenizer.tokenize(text):
              for sub_token in self.wordpiece_tokenizer.tokenize(token):
                  split_tokens.append(sub_token)
        else:
          split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        #pretrained_model_name_or_path : bert-base-uncased
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        #vocab_file : .torch_pretrained_bert/bert-base-uncased-vocab.txt
        
        # false
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        try:
            # .torch_pretrained_bert/bert-base-uncased-vocab.txt
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        
        #True
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        
        #True
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            #512
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            #512
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
            
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False



# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""
from collections import namedtuple
import random
import os
import csv
import itertools

import nltk
from nltk import tokenize as nltk_tokenize
import sentencepiece as spm

import regex as re





def make_tokenizer(tokenizer_type, corpus, model_path=None, vocab_size=None, model_type=None, pad_token=0,
                   character_coverage=1.0, command_tokens=None, type_tokens=None, **kwargs):
    #BertWordPieceTokenizer
    tokenizer_class = tokenizer_type
    #True
    if isinstance(tokenizer_class, str):
        tokenizer_class = eval(tokenizer_class)
    
    #True
    if tokenizer_class is BertWordPieceTokenizer:
        #here
        return BertWordPieceTokenizer(model_type, **kwargs)
    text_tokenizer = tokenizer_class(corpus=corpus, vocab_size=vocab_size, model_path=model_path, model_type=model_type,
                                     pad_token=pad_token, character_coverage=character_coverage)
    return Tokenizer(text_tokenizer, command_tokens, type_tokens)


class Tokenization(object):
    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def insert(self, idx, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization) - 1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, (CommandToken, TypeToken)):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], (CommandToken, TypeToken)):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


DEFAULT_COMMAND_TOKENS = [
    ('pad', 0),
    ('eos', 1),
    ('bos', 2),
    ('unk', 3),
    ('sep', 4),
    ('L2R', 5),
    ('ENC', 6),
    ('MASK', 7),
]
DEFAULT_COMMAND_TOKENS = prep_command_tokens(DEFAULT_COMMAND_TOKENS)

TYPE_TUPLE = namedtuple('TypeToken', ('name', 'token', 'Id'))


def prep_type_tokens(tokenlist, token_format=token_format):
    return [TypeToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


DEFAULT_TYPE_TOKENS = [
    ('function', 0),
    ('command', 1),
    ('str0', 2),
    ('str1', 3),
    ('str2', 4),
    ('embedding0', 5),
    ('embedding1', 6),
    ('embedding2', 7),
    ('arg0', 8),
    ('arg1', 9),
    ('arg2', 10),
]
DEFAULT_TYPE_TOKENS = prep_type_tokens(DEFAULT_TYPE_TOKENS)


class Tokenizer(object):
    def __init__(self, text_tokenizer, command_tokens=None, type_tokens=None):
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)

        if command_tokens is None:
            command_tokens = DEFAULT_COMMAND_TOKENS
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}
        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self._command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_command_tokens + self.num_text_tokens

        if type_tokens is None:
            type_tokens = DEFAULT_TYPE_TOKENS
        self.type_tokens = type_tokens
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}
        if not hasattr(self, 'num_type_tokens'):
            self.num_type_tokens = len(self.type_tokens)

        self._tokens = list(self.command_token_map.keys()) + list(self.text_tokenizer.tokens)
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        return self.num_tokens

    def get_command(self, name):
        return self.command_name_map[name]

    def get_type(self, name):
        return self.type_name_map[name]

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    @property
    def token_types(self):
        return self._token_types

    @property
    def token_type_vocab(self):
        return self._token_type_vocab

    @property
    def command_tokens(self):
        return self._command_token_tokens

    @property
    def command_token_vocab(self):
        return self._command_token_vocab

    @property
    def text_tokens(self):
        return self._text_tokens

    @property
    def text_token_vocab(self):
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if tok_extended.rstrip and i > 0:
                    sub_text = sub_text.lstrip()
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip() 

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def EncodeAsTokens(self, text, process_fn=None):
        tokenization = self.text_tokenizer.EncodeAsTokens(text, process_fn=process_fn)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id < self.num_command_tokens:
            return self.command_id_map[Id].token
        return self.text_tokenizer.IdToToken(Id - self.num_command_tokens)

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        if token in self.command_token_map:
            return self.command_token_map[token].Id
        return self.text_tokenizer.TokenToId(token) + self.num_command_tokens

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id < self.num_command_tokens:
                rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id - self.num_command_tokens)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeIds(current_str))
        return ' '.join(rtn_strs)

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        rtn_strs = []
        current_str = []
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        for t in Tokens:
            if isinstance(t, CommandToken):
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t.token)
            elif t in self.command_token_map:
                rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
                current_str = []
                rtn_strs.append(t)
            else:
                current_str.append(t)
        if current_str != []:
            rtn_strs.append(self.text_tokenizer.DecodeTokens(current_str))
        return ' '.join(rtn_strs)


class TextTokenizer(object):
    """
    Interface for text tokenizer
    """

    def __init__(self):
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = 0
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = self.num_text_tokens

    def __call__(self, text, process_fn=None):
        return self.EncodeAsIds(text, process_fn)

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        raise NotImplementedError('TextTokenizer tokens property not implemented')

    @property
    def vocab(self):
        """dictionary mapping tokens to ids"""
        raise NotImplementedError('TextTokenizer vocab property not implemented')

    @staticmethod
    def exists(model_path):
        """check if the filepath for a text tokenizer exists"""
        raise NotImplementedError('TextTokenizer exists method not implemented')

    def Train(self, corpus):
        """train a tokenizer on a data corpus and save model for future use"""
        raise NotImplementedError('TextTokenizer Train not implemented')

    def EncodeAsIds(self, text, process_fn=None):
        """
        Preprocess text and encode as ids. Return a tokenization object with
        original text, processed text, and id tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsIds not implemented')

    def EncodeAsTokens(self, text, process_fn=None):
        """
        Preprocess text and encode as tokens. Return a tokenization object with
        original text, processed text, and token tokenization.
        """
        raise NotImplementedError('TextTokenizer EncodeAsTokens not implemented')

    def IdToToken(self, Id):
        """Convert an Id to Token. Reverse lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer IdToToken not implemented')

    def TokenToId(self, token):
        """Convert a Token to Id. Lookup of self.vocab"""
        raise NotImplementedError('TextTokenizer TokenToId not implemented')

    def DecodeIds(self, Ids):
        """Convert a list or tokenization object of Ids to a text string"""
        raise NotImplementedError('TextTokenizer DecodeIds not implemented')

    def DecodeTokens(self, Tokens):
        """Convert a list or tokenization object of tokens to a text string"""
        raise NotImplementedError('TextTokenizer DecodeTokens not implemented')


class CharacterLevelTokenizer(TextTokenizer):
    """
    Text tokenizer for ASCII-256 Character Level Tokenization.
    """

    def __init__(self, **kwargs):
        self.num_text_tokens = 256
        super(CharacterLevelTokenizer, self).__init__()
        self._tokens = [self.IdToToken(Id) for Id in range(self.num_text_tokens)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    def __len__(self):
        return 256

    @staticmethod
    def exists(model_path):
        return True

    def Train(self, corpus):
        pass

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to ascii 256 Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
            processed_text = str(processed_text)
        tokens = [self.TokenToId(c) for c in processed_text]
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to ascii 256 characters"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        processed_text = str(processed_text)
        tokens = [c for c in processed_text]
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """ascii index to character"""
        return chr(Id)

    def TokenToId(self, token):
        """ascii character to index"""
        return ord(token)

    def DecodeIds(self, Ids):
        """converts ascii ids to tokens before joining them into text"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return ''.join([self.IdToToken(tok) for tok in Ids])

    def DecodeTokens(self, Tokens):
        """just concatenates ascii tokens into text"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ''.join(Tokens)


MAX_SENTENCEPIECE_SENTENCES = 100000000


def get_corpus_freq(dataset, filepath, filetype='tsv'):
    """
    Take corpus, split it into sentences, and extract word frequencies.
    Write frequencies to `filepath` as a tsv. Only write the first
    MAX_SENTENCEPIECE_SENTENCES most common words to the file.
    """
    nltk.download('punkt', download_dir="./nltk")
    if filetype == 'tsv':
        delimiter = '\t'
    else:
        delimiter = ','

    print("compute corpus frequency\n", flush=True)

    total_sentence_count = 0
    maxlen = 0
    freqs = {}
    for entry in dataset:
        if isinstance(entry, dict):
            entry = entry['text']
        lines = entry.strip().split('\n')
        for line in lines:
            sentences = nltk_tokenize.sent_tokenize(line)
            total_sentence_count += len(sentences)
            for sentence in sentences:
                maxlen = max(len(line), maxlen)
                for word in sentence.split():
                    if word not in freqs:
                        freqs[word] = 0
                    freqs[word] += 1

    print("length of freqs before truncating " + str(len(freqs)), flush=True)
    print("file path for freq " + str(filepath), flush=True)

    freqs_sorted = {}
    counter = 0
    for word, count in sorted(freqs.items(), key=lambda x: x[1], reverse=True):
        if counter >= MAX_SENTENCEPIECE_SENTENCES:
            break
        counter += 1
        freqs_sorted[word] = count

    print("length of freqs after trancating " + str(len(freqs_sorted)), flush=True)

    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        for k, v in freqs_sorted.items():
            writer.writerow([str(k), str(v)])

    return total_sentence_count, maxlen


class SentencePieceTokenizer(TextTokenizer):
    """Trains and uses sentencepiece for text tokenization"""

    def __init__(self, model_type='bpe', vocab_size=None, corpus=None, model_path=None, character_coverage=1.0,
                 **kwargs):
        self.character_coverage = character_coverage
        self.model_type = model_type.lower()
        self.spm_model = model_path
        self.num_text_tokens = vocab_size
        make_train = not SentencePieceTokenizer.exists(self.spm_model)
        if make_train:
            assert corpus is not None and self.num_text_tokens is not None
            self.Train(corpus, self.num_text_tokens)
        self._tokens = []
        self._vocab = {}
        self.load_spm_model()
        super(SentencePieceTokenizer, self).__init__()

    def __len__(self):
        return self.num_text_tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def exists(model_path):
        if model_path is None:
            return False
        # check if path exists
        dne = not os.path.exists(model_path)
        # check if path.model exists
        if dne and not model_path.endswith('.model'):
            dne = not os.path.exists(model_path + '.model')
        return not dne

    def load_spm_model(self):
        """load sentencepiece model and parse vocab"""
        if not os.path.exists(self.spm_model) and not self.spm_model.endswith('.model'):
            self.spm_model = self.spm_model + '.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model)
        self.vocab_size = self.num_text_tokens = len(self.sp)
        self._tokens = [self.IdToToken(t) for t in range(self.vocab_size)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    def Train(self, corpus, num_text_tokens):
        """train sentencepiece model on corpus using word frequencies"""
        self.num_text_tokens = num_text_tokens
        use_model_path = self.spm_model
        random_hash = str(random.randint(0, 2147483647))
        if use_model_path is None:
            use_model_path = random_hash
        if use_model_path.endswith('.model'):
            use_model_path = use_model_path[:use_model_path.rfind('.model')]
        input_path = use_model_path + '.tsv.' + random_hash
        line_count, maxlenline = get_corpus_freq(corpus, input_path)
        line_count = min(line_count, MAX_SENTENCEPIECE_SENTENCES)
        print('line count used as input_sentence_size ', line_count, flush=True)
        print('training sentencepiece model', flush=True)
        train_string = '--input={file_path} --model_prefix={model_prefix} --vocab_size={vocab_size}' \
                       + ' --model_type={model_type} --character_coverage={character_coverage} ' \
                       + '--input_sentence_size={input_sentence_size} ' \
                       + '--input_format=tsv'
        train_string = train_string.format(file_path=input_path, model_prefix=use_model_path,
                                           vocab_size=num_text_tokens,
                                           model_type=self.model_type, character_coverage=self.character_coverage,
                                           input_sentence_size=int(line_count))  # , #)#,
        print("calling spm.SentencePieceTrainer.Train(%s)" % (train_string), flush=True)
        spm.SentencePieceTrainer.Train(train_string)
        os.remove(input_path)
        self.spm_model = use_model_path + '.model'
        print('sentencepiece model written to ' + self.spm_model, flush=True)

    def EncodeAsIds(self, text, process_fn=None):
        """convert text to sentencepiece Ids"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsIds(processed_text)
        return Tokenization(tokens, processed_text, text)

    def EncodeAsTokens(self, text, process_fn=None):
        """convert text to sentencepiece tokens"""
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.sp.EncodeAsTokens(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id):
        """convert Id to sentencpiece token"""
        return self.sp.IdToPiece(Id)

    def TokenToId(self, token):
        """convert sentencpiece token to Id"""
        return self.sp.PieceToId(token)

    def DecodeIds(self, Ids):
        """converts ids to a text string"""
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        return self.sp.DecodeIds(Ids)

    def DecodeTokens(self, Tokens):
        """converts sentencepiece tokens to a text string"""
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return self.sp.DecodeTokens(Tokens)


class BertWordPieceTokenizer(Tokenizer):
    def __init__(self, tokenizer_model_type=None, cache_dir=None, add_block_symbols=False, add_sentinel_token=0,
                 add_task_mask=False, add_decoder_mask=False, **kwargs):
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'   #bert-base-uncased

        #True
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('loading BertWordPieceTokenizer (', tokenizer_model_type, ') from cache_dir ', cache_dir)

        #True
        do_lower_case = not ('-cased' in tokenizer_model_type or 'chinese' in tokenizer_model_type)
    
        self.text_tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type, do_lower_case=do_lower_case,
                                                            cache_dir=cache_dir)
        #True
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('loaded', tokenizer_model_type)

        self.text_tokenizer.max_len = int(1e12)
        self.num_command_tokens = 6
        self.num_tokens = len(self.text_tokenizer.vocab)
        self.num_text_tokens = self.num_tokens - 5
        self.num_type_tokens = 2

        self._command_tokens = [
            CommandToken('pad', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
            CommandToken('ENC', '[CLS]', self.text_tokenizer.vocab['[CLS]']),
            CommandToken('MASK', '[MASK]', self.text_tokenizer.vocab['[MASK]']),
            CommandToken('unk', '[UNK]', self.text_tokenizer.vocab['[UNK]']),
            CommandToken('sep', '[SEP]', self.text_tokenizer.vocab['[SEP]']),
            CommandToken('eos', '[PAD]', self.text_tokenizer.vocab['[PAD]']),
        ]
        if add_block_symbols:
            self._command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', self.num_tokens),
                CommandToken('eop', '<|endofpiece|>', self.num_tokens + 1)
            ])
            self.num_tokens += 2
            self.num_command_tokens += 2
            if add_task_mask:
                self._command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', self.num_tokens),
                    CommandToken('sMASK', '[sMASK]', self.num_tokens + 1)
                ])
                self.num_tokens += 2
                self.num_command_tokens += 2
            if add_decoder_mask:
                self._command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', self.num_tokens)
                ])
                self.num_tokens += 1
                self.num_command_tokens += 1
        if add_sentinel_token > 0:
            for i in range(1, add_sentinel_token):
                self._command_tokens.extend([CommandToken(f'MASK{i}', f'[MASK{i}]', self.num_tokens),
                                             CommandToken(f'sop{i}', f'<|startofpiece{i}|>', self.num_tokens + 1)])
                self.num_tokens += 2
                self.num_command_tokens += 2
        self.command_name_map = {tok.name: tok for tok in self._command_tokens}
        self.command_token_map = {tok.token: tok for tok in self._command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self._command_tokens}

        self.type_tokens = [
            TypeToken('str0', '<str0>', 0),
            TypeToken('str1', '<str1>', 1),
        ]
        self.type_name_map = {tok.name: tok for tok in self.type_tokens}
        self.type_token_map = {tok.token: tok for tok in self.type_tokens}
        self.type_id_map = {tok.Id: tok for tok in self.type_tokens}

        self._tokens = list(self.text_tokenizer.vocab.keys())
        self._vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._text_tokens = list(self._tokens)
        self._text_token_vocab = {k: v for k, v in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self._token_types = list(self.type_token_map.keys())
        self._token_type_vocab = {t: Id for Id, t in self.type_id_map.items()}

    def _encode(self, text):
        tokens = self.text_tokenizer.tokenize(text)
        ids = self.text_tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def EncodeAsTokens(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        tokens = self.text_tokenizer.tokenize(processed_text)
        return Tokenization(tokens, processed_text, text, asIds=False)

    def IdToToken(self, Id, type_token=False):
        if isinstance(Id, (TypeToken, CommandToken)):
            return Id.token
        if type_token:
            return self.type_id_map[Id].token
        if Id in self.command_id_map:
            return self.command_id_map[Id].token
        return self.text_tokenizer.ids_to_tokens[Id]

    def TokenToId(self, token, type_token=False):
        if isinstance(token, (TypeToken, CommandToken)):
            return token.Id
        if type_token:
            return self.type_token_map[token].Id
        return self.text_tokenizer.vocab[token]

    def DecodeIds(self, Ids, type_token=False):
        if type_token:
            return ' '.join(Id.token if isinstance(Id, TypeToken) else self.type_id_map[Id].token for Id in Ids)
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        Tokens = []
        for Id in Ids:
            if Id in self.command_id_map:
                Tokens.append(self.command_id_map[Id].token)
            elif Id in self.text_tokenizer.ids_to_tokens:
                Tokens.append(self.text_tokenizer.ids_to_tokens[Id])
        new_tokens = []
        for token in Tokens:
            if token.startswith('##') and len(new_tokens) > 0:
                new_tokens[-1] += token[2:]
            else:
                new_tokens.append(token)
        return ' '.join(new_tokens)

    def DecodeTokens(self, Tokens, type_token=False):
        if type_token:
            return ' '.join(t.token if isinstance(t, TypeToken) else t for t in Tokens)
        if isinstance(Tokens, Tokenization):
            Tokens = Tokens.tokenization
        return ' '.join(Tokens)

