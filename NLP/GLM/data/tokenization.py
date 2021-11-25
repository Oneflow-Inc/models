from collections import namedtuple
import collections
import itertools
import logging
import os
import unicodedata
from .corpora import print_rank_0
from .file_utils import cached_path

logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': ".oneflow_pretrained_bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': ".oneflow_pretrained_bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': ".oneflow_pretrained_bert/bert-base-cased-vocab.txt",
    'bert-large-cased': ".oneflow_pretrained_bert/bert-large-cased-vocab.txt",
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


def make_tokenizer(tokenizer_type, corpus, model_path=None, vocab_size=None, model_type=None, pad_token=0,
                   character_coverage=1.0, command_tokens=None, type_tokens=None, **kwargs):
    # BertWordPieceTokenizer
    tokenizer_class = tokenizer_type
    # True
    if isinstance(tokenizer_class, str):
        tokenizer_class = eval(tokenizer_class)

    # True
    if tokenizer_class is BertWordPieceTokenizer:
        # here
        return BertWordPieceTokenizer(model_type, **kwargs)
    else:
        raise NotImplementedError


def prepare_tokenizer(args):
    add_sentinel_token = 0

    # False
    if args.sentinel_token:
        add_sentinel_token = args.max_position_embeddings  # 0
    # here
    tokenizer = make_tokenizer(args.tokenizer_type, None, args.tokenizer_path, args.vocab_size,
                               args.tokenizer_model_type, add_block_symbols=args.block_lm, cache_dir=args.cache_dir,
                               add_sentinel_token=add_sentinel_token, add_task_mask=args.task_mask,
                               add_decoder_mask=args.block_mask_prob > 0.0 or args.context_mask_ratio > 0.0)
    num_tokens = tokenizer.num_tokens
    eod_token = tokenizer.get_command('eos').Id
    assert eod_token == tokenizer.get_command('pad').Id
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(before, after - before, after))
    print_rank_0('> found end-of-document token: {}'.format(eod_token))
    # token_counts = torch.cuda.LongTensor([after, eod_token])
    # else:
    #     token_counts = torch.cuda.LongTensor([0, 0])
    num_tokens = after
    eod_token = eod_token
    args.vocab_size, args.eod_token = num_tokens, eod_token
    return tokenizer


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
            self.tokenization = self.tokenization[:idx] + \
                other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + \
                other.tokenization + self.tokenization[idx:]

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


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __str__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


class TypeToken(object):
    def __init__(self, name, token, Id):
        self.name = name
        self.token = token
        self.Id = Id

    def __str__(self):
        return str(TYPE_TUPLE(self.name, self.token, self.Id))


token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


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
        self.command_token_map = {
            tok.token: tok for tok in self._command_tokens}
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

        self._tokens = list(self.command_token_map.keys()) + \
            list(self.text_tokenizer.tokens)
        self._vocab = {t: Id for Id, t in self.command_id_map.items()}
        self._vocab.update({t: Id + self.num_command_tokens for t,
                           Id in self.text_tokenizer.vocab.items()})

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {
            t: Id + self.num_command_tokens for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id for Id, t in self.command_id_map.items()}

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
        tokenization = self.text_tokenizer.EncodeAsTokens(
            text, process_fn=process_fn)
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


def load_vocab(vocab_file):
    # .oneflow_pretrained_bert/bert-base-uncased-vocab.txt
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


def _is_whitespace(char):
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
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


def _is_control(char):
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def whitespace_tokenize(text):
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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


class BertTokenizer(object):

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        # False
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
                " sequence through BERT will result in indexing errors".format(
                    len(ids), self.max_len)
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
        # vocab_file : .oneflow_pretrained_bert/bert-base-uncased-vocab.txt

        # false
        if os.path.isdir(vocab_file):
            vocab_file = os.path.join(vocab_file, VOCAB_NAME)
        try:
            # .oneflow_pretrained_bert/bert-base-uncased-vocab.txt
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

        # True
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))

        # True
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # 512
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            # 512
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)

        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        
        return tokenizer


class BertWordPieceTokenizer(Tokenizer):
    def __init__(self, tokenizer_model_type=None, cache_dir=None, add_block_symbols=False, add_sentinel_token=0,
                 add_task_mask=False, add_decoder_mask=False, **kwargs):
        if tokenizer_model_type not in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tokenizer_model_type = 'bert-large-uncased'  # bert-base-uncased

        # True
        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('loading BertWordPieceTokenizer (', tokenizer_model_type, ') from cache_dir ', cache_dir)

        # True
        do_lower_case = not (
            '-cased' in tokenizer_model_type or 'chinese' in tokenizer_model_type)

        self.text_tokenizer = BertTokenizer.from_pretrained(tokenizer_model_type, do_lower_case=do_lower_case,
                                                            cache_dir=cache_dir)
        # True
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
            CommandToken('MASK', '[MASK]',
                         self.text_tokenizer.vocab['[MASK]']),
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
        self.command_token_map = {
            tok.token: tok for tok in self._command_tokens}
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
        self._text_token_vocab = {k: v for k,
                                  v in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {
            t: Id for Id, t in self.command_id_map.items()}

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
