import oneflow as flow
from oneflow.utils import data
import copy
import random
import numpy as np
from operator import itemgetter
from bisect import bisect_right
from . import corpora
from .corpora import print_rank_0
from .lazy_loader import LazyLoader, LazyWriter
from .blocklm import ConstructBlockStrategy


def should_split(split):
    return max(split) / sum(split) != 1.


def get_split(args):
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1 - split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s / final_sum for s in splits]


class ConcatDataset(data.Dataset):

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.is_lazy = sum([isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy') and ds.is_lazy) for ds in
                            self.datasets]) == len(self.datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._X = None
        self._Y = None
        self._lens = None

    def get_text_len(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_text_len(sample_idx)

    def SetTokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.datasets[0].GetTokenizer()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def lens(self):
        if self._lens is None:
            self._lens = []
            if self.is_lazy:
                for data in self.datasets:
                    self._lens.extend(data.lens)
            else:
                for data in self.datasets:
                    self._lens.extend([len(d['text']) if isinstance(
                        d, dict) else len(d) for d in data])
        return self._lens

    @property
    def X(self):
        if self._X is None:
            self._X = []
            for data in self.datasets:
                self._X.extend(data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = []
            for data in self.datasets:
                self._Y.extend(list(data.Y))
            self._Y = np.array(self._Y)
        return self._Y


class SplitDataset(data.Dataset):

    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.wrapped_data = ds
        self.is_lazy = isinstance(ds, LazyLoader) or (
            hasattr(ds, 'is_lazy') and ds.is_lazy)
        self._X = None
        self._Y = None

    def __len__(self):
        return len(self.split_inds)

    def get_text_len(self, idx):
        return self.wrapped_data.get_text_len(self.split_inds[idx])

    def __getitem__(self, index):
        return self.wrapped_data[self.split_inds[index]]

    def SetTokenizer(self, tokenizer):
        self.wrapped_data.SetTokenizer(tokenizer)

    def GetTokenizer(self):
        return self.wrapped_data.GetTokenizer()

    @property
    def X(self):
        if self._X is None:
            self._X = itemgetter(*self.split_inds)(self.wrapped_data.X)
        return self._X

    @property
    def Y(self):
        if self._Y is None:
            self._Y = np.array(itemgetter(*self.split_inds)
                               (self.wrapped_data.Y))
        return self._Y

    def __iter__(self):
        for idx in self.split_inds:
            yield self.wrapped_data[idx]


def split_ds(ds, split=None, shuffle=True, save_splits=None, load_splits=None):

    if split is None:
        split = [.8, .2, .0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    ds_len = len(ds)
    inds = np.arange(ds_len)
    if shuffle:
        rng = np.random.RandomState(1234)
        rng.shuffle(inds)
    if load_splits is not None:
        inds = np.load(load_splits)
        assert len(inds) == ds_len
        print_rank_0(f"Load split indices from {load_splits}")
    elif save_splits is not None:
        if flow.distributed.get_rank() == 0:
            np.save(save_splits, inds)
            print(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = ds_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = inds[start_idx:start_idx + max(split_, 1)]
            rtn_ds[i] = SplitDataset(ds, split_inds)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


def get_dataset(path, seq_length, mem_length, shuffle=True, split=None, tokenizer=None,
                sample_one_document=False, pre_tokenize=False, ds_type='', save_splits=None, load_splits=None,
                save_test_data=None, no_lazy_loader=False, loader_scatter=None, data_parallel_rank=None,
                filter_english=False, non_sentence_start=0.0, half_lazy_loader=False, **kwargs):

    assert len(path) == 1

    name = path[0]

    dataset = corpora.NAMED_CORPORA[name]
    path = dataset.PATH

    map_fn = (lambda x: x.tolist()) if pre_tokenize else None
    prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                         is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
    texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                       is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)

    text_dataset = corpora.PromptDataset(prompt_loader=prompts, text_loader=texts, tokenizer=tokenizer,
                                         to_tokenize=not pre_tokenize)

    if loader_scatter is None:
        for i in range(10):
            rand_id = i if i < 5 else random.randrange(len(text_dataset))
            sample_tokens = text_dataset[rand_id]['tokens'][:1024]
            print(sample_tokens)
            print(tokenizer.DecodeIds(sample_tokens).encode('utf-8'))
    else:
        raise NotImplementedError

    ds = ConcatDataset([text_dataset])
    if should_split(split):
        ds = split_ds(ds, split, shuffle=shuffle,
                      save_splits=save_splits, load_splits=load_splits)
        ds = [corpora.BlockDataset(d, tokenizer,
                                   max_seq_len=seq_length,
                                   sample_across_doc=not sample_one_document,
                                   filter_english=filter_english,
                                   non_sentence_start=non_sentence_start)
              if d is not None else None for d in ds]

    return ds


def make_dataset(args, tokenizer):
    world_size = 1

    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    # eval_batch_size:16

    seq_length = args.seq_length
    if seq_length < 0:
        seq_length = seq_length * world_size
    # seq_length:512

    eval_seq_length = args.eval_seq_length
    if eval_seq_length is not None and eval_seq_length < 0:
        eval_seq_length = eval_seq_length * world_size
    # eval_seq_length:none

    split = get_split(args)

    data_set_args = {
        'path': args.train_data,
        'seq_length': seq_length,
        'mem_length': args.mem_length,
        'delim': args.delim,
        'text_key': args.text_key,
        'label_key': 'label',
        'ds_type': args.data_set_type,
        'split': split,
        'loose': args.loose_json,
        'max_preds_per_seq': args.max_preds_per_seq,
        'presplit_sentences': args.presplit_sentences,
        'sample_one_document': args.sample_one_document,
        'filter_english': args.filter_english,
        'pre_tokenize': not args.no_pre_tokenize,
        'tokenizer': tokenizer,
        'save_splits': args.save_splits,
        'load_splits': args.load_splits,
        'save_test_data': args.save_test_data,
        'no_lazy_loader': args.no_lazy_loader,
        'loader_scatter': args.loader_scatter,
        'data_parallel_rank': 0,
        "non_sentence_start": args.non_sentence_start,
        "half_lazy_loader": args.half_lazy_loader
    }

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]

    # None
    if eval_seq_length:
        eval_set_args['seq_length'] = eval_seq_length
    # None
    if args.eval_max_preds_per_seq:
        eval_set_args['max_preds_per_seq'] = args.eval_max_preds_per_seq
    # False
    if args.eval_text_key is not None:
        eval_set_args['text_key'] = args.eval_text_key

    train, valid, test = None, None, None

    # True
    if args.train_data is not None:
        train = get_dataset(**data_set_args)
        # True
        if should_split(split):
            train, valid, test = train
        eval_set_args['tokenizer'] = tokenizer

    # False
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid = get_dataset(**eval_set_args)
        eval_set_args['tokenizer'] = tokenizer

    # False
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test = get_dataset(**eval_set_args)

    use_block = args.block_lm or args.encoder_decoder

    # true
    if train is not None and args.batch_size > 0:
        train = make_data_loader(train, tokenizer, batch_size, args, shuffle=args.shuffle,
                                 block_collate=use_block)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size

    # True
    if valid is not None:
        valid = make_data_loader(valid, tokenizer, eval_batch_size, args, shuffle=args.shuffle,
                                 block_collate=use_block)
        args.do_valid = True
    else:
        args.do_valid = False

    # True
    if test is not None:
        test = make_data_loader(test, tokenizer, eval_batch_size, args,
                                shuffle=args.shuffle, block_collate=use_block)
        args.do_test = True
    else:
        args.do_test = False

    return train, valid, test


def make_data_loader(dataset, tokenizer, batch_size, args, shuffle=False, block_collate=False):
    if shuffle:
        raise NotImplementedError
        # sampler = data_utils.samplers.RandomSampler(dataset, replacement=True,
        #                                             num_samples=batch_size * args.train_iters * args.gradient_accumulation_steps)
    else:
        sampler = flow.utils.data.SequentialSampler(dataset)
    drop_last = False
    # False
    if drop_last:
        raise NotImplementedError
        # batch_sampler = data_utils.samplers.DistributedBatchSampler(sampler, batch_size, drop_last, rank,
        #                                                             world_size,
        #                                                             gradient_accumulation_steps=args.gradient_accumulation_steps)
    else:
        batch_sampler = flow.utils.data.BatchSampler(sampler,
                                                     batch_size,
                                                     drop_last)
    collate_fn = None
    # True
    if block_collate:
        collate_fn = ConstructBlockStrategy(args, tokenizer, args.seq_length,
                                            bert_prob=args.bert_prob,
                                            gap_sentence_prob=args.gap_sentence_prob,
                                            gap_sentence_ratio=args.gap_sentence_ratio,
                                            gpt_infill_prob=args.gpt_infill_prob,
                                            average_block_length=args.avg_block_length,
                                            gpt_min_ratio=args.gpt_min_ratio,
                                            block_mask_prob=args.block_mask_prob,
                                            context_mask_ratio=args.context_mask_ratio,
                                            short_seq_prob=args.short_seq_prob,
                                            single_span_prob=args.single_span_prob,
                                            shuffle_blocks=not args.no_shuffle_block,
                                            block_position_encoding=not args.no_block_position,
                                            sentinel_token=args.sentinel_token,
                                            encoder_decoder=args.encoder_decoder,
                                            task_mask=args.task_mask,
                                            random_position=args.random_position,
                                            masked_lm=args.masked_lm).construct_blocks

    data_loader = flow.utils.data.DataLoader(dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn)
    return data_loader
