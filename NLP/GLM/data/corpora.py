import os
import json
import tqdm
import random
import numpy as np
from bisect import bisect_right
from itertools import accumulate
from multiprocessing import Queue, Process
from oneflow.utils import data
import oneflow as flow
from .lazy_loader import LazyLoader

def print_rank_0(message):
    if flow.env.get_rank():
        print(message, flush=True)


NUM_PROCESSES = 100

class PromptDataset(data.Dataset):
    def __init__(self, prompt_loader, text_loader, tokenizer=None, to_tokenize=False, **kwargs):
        self.prompts = prompt_loader
        self.texts = text_loader
        self.tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if isinstance(self.prompts, LazyLoader) and isinstance(self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        return self.prompt_lens[idx] + self.text_lens[idx]

    def __getitem__(self, index):
        prompt = self.prompts[index]
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt).tokenization
            text = self.tokenizer.EncodeAsIds(text).tokenization
        return {"tokens": prompt + text, "loss_masks": [0] * len(prompt) + [1] * len(text)}

    def __len__(self):
        return len(self.prompts)



def punctuation_standardization(string: str):
    punctuation_dict = {"\u201c": "\"", "\u201d": "\"",
                        "\u2019": "'", "\u2018": "'", "\u2013": "-"}
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


class DataReader:
    PATH = None
    assert_str = None
    reserve_punct = False

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        raise NotImplementedError

    def print_info(self, info):
        pass

    def __init__(self, writers, tokenizer=None, tokenize=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        print_rank_0(f"Creating dataset from {self.PATH}")
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writers = writers

    def process(self):
        if os.path.isdir(self.PATH):
            paths = [entry.path for entry in os.scandir(self.PATH) if
                     not entry.is_dir() and not entry.name.endswith("bz2")]
        else:
            paths = [self.PATH]
        task_queue, done_queue, info_queue = Queue(
            maxsize=10000000), Queue(maxsize=10000000), Queue()
        processes = []
        for i in range(NUM_PROCESSES):
            process = Process(target=self.tokenize_worker,
                              args=(task_queue, done_queue, info_queue, self.tokenizer, self.tokenize))
            process.start()
            processes.append(process)

        def read_input_to_queue():
            for path in paths:
                print_rank_0(f"Start reading {path}")
                with open(path) as file:
                    for row in file:
                        task_queue.put(row)
            print_rank_0("Read input complete")
            for i in range(len(processes)):
                task_queue.put('STOP')

        process = Process(target=read_input_to_queue)
        process.start()
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writers)
                progress_bar.update()
        progress_bar.close()
        self.print_info(info_queue)

    @staticmethod
    def write_result(data, writers):
        raise NotImplementedError

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @classmethod
    def process_sample(cls, text, tokenizer, tokenize):
        # True
        if isinstance(text, str) and tokenize:
            # True
            if not cls.reserve_punct:
                text = punctuation_standardization(text)
            text = tokenizer.EncodeAsIds(text).tokenization if text else []
        return text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += "......"
        return content

    def process_line(self, data, tokenizer, tokenize):
        raise NotImplementedError


class PromptReader(DataReader):
    is_json = True

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            row = row.rstrip()
            if row:
                if self.is_json:
                    row = json.loads(row)
                prompts, texts = self.process_line(row, tokenizer, tokenize)
                for prompt, text in zip(prompts, texts):
                    output.put((prompt, text))
        output.put("COMPLETE")

    @staticmethod
    def write_result(data, writers):
        prompt, text = data
        writers['prompt'].write(prompt)
        writers['text'].write(text)


class BertData(PromptReader):
    is_json = False
    PATH = '/dataset/fd5061f6/english_data/wikibook'

    def process_line(self, data, tokenizer, tokenize):
        if data:
            prompt, text = "", data
            prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
            return [prompt], [text]
        else:
            return [], []

#


class BertLargeData(BertData):
    #PATH = '/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large'
    PATH = './other/dataset/glm_train.txt'


NAMED_CORPORA = {
    'wikibook': BertData,
    "bert-large": BertLargeData,
}


class BlockDataset(data.Dataset):
    def __init__(self, ds, tokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start=0.0, filter_english=False, **kwargs):
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start = non_sentence_start
        self.filter_english = filter_english
        self.weighting, self.total_len = None, None
        self.is_lazy = False
        if self.filter_english:
            import fasttext
            self.model = fasttext.load_model('/mnt/lid.176.bin')
            print_rank_0("Load language detection model")
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.is_lazy:
            lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([len(d['text']) if isinstance(d, dict) else len(d) for d in self.ds])
        self.total_len = np.sum(lens)
        print_rank_0(
            f"Dataset document count {len(lens)}, token count {self.total_len}, non sentence start{self.non_sentence_start}")
        self.weighting = list(accumulate(lens))

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.total_len)
            data_idx = bisect_right(self.weighting, idx)
            tokens, loss_mask = self.getidx(data_idx)
            if self.filter_english:
                text = self.tokenizer.DecodeIds(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        tokens, loss_mask = self.get_weighted_samples(rng)
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(
                            tokens[strip_left_tokens - 1]):
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.get_command('eos').Id:
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask, self.max_seq_len)
        else:
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens
            loss_mask = [0] + loss_mask

            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.get_command('ENC').Id] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(new_tokens, new_loss_mask,
                                                                     self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(tokens) - 1 and not self.contains_sentence_end(
                    tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        if '\n' in tok:
            return True
        return False
