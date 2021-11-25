import os
import mmap
import pickle as pkl
import time
import numpy as np
from itertools import accumulate

import oneflow as flow
from oneflow.multiprocessing import Lock


__all__ = [
    "LazyWriter",
    "LazyLoader",
]

def get_lazy_path(path):
    return os.path.splitext(path)[0] + '.lazy'


def exists_lazy(path, data_type='data'):
    if not os.path.exists(get_lazy_path(path)):
        return False
    contents = os.listdir(get_lazy_path(path))
    if data_type not in contents:
        return False
    if data_type + '.len.pkl' not in contents:
        return False
    return True


def get_scatter_path(path, scatter_rank):
    path = os.path.splitext(path)[0] + '.scatter'
    scatter_path = os.path.join(path, str(scatter_rank))
    return scatter_path


def exists_scatter(path, scatter_num=64, data_type='data'):
    for i in range(scatter_num):
        scatter_path = get_scatter_path(path, scatter_rank=i)
        if not exists_lazy(scatter_path, data_type=data_type):
            return False
    return True


class LazyWriter:
    def __init__(self, path, data_type, is_array=False, array_data_type=np.int32):
        # path:/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large
        # lazypath:/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large.lazy
        lazypath = get_lazy_path(path)
        if not os.path.exists(lazypath):
            os.makedirs(lazypath)
        self.datapath = os.path.join(lazypath, data_type)
        self.lenpath = os.path.join(lazypath, data_type + '.len.pkl')
        self.array_data_type = array_data_type
        self.output = open(self.datapath, 'wb')
        self.lengths = []
        self.is_array = is_array

    @staticmethod
    def get_len_path(path, data_type):
        lazypath = get_lazy_path(path)
        return os.path.join(lazypath, data_type + '.len.pkl')

    def write(self, s):
        if isinstance(s, dict):
            s = s['text']
        if self.is_array:
            encoded = np.array(
                s, dtype=self.array_data_type).tobytes(order='C')
            self.output.write(encoded)
            self.lengths.append(len(s))
        else:
            encoded = s.encode('utf-8')
            self.output.write(encoded)
            self.lengths.append(len(encoded))

    def close(self):
        self.output.close()
        with open(self.lenpath, 'wb') as f:
            pkl.dump(self.lengths, f)


def split_strings(strings, start, chr_lens):
    return [strings[i - start:j - start] for i, j in zip([start] + chr_lens[:-1], chr_lens)]


class ProcessorTokenizer:
    def __init__(self, tokenizer, process_fn=None):
        self.tokenizer = tokenizer
        self.process_fn = process_fn

    def __call__(self, string):
        if self.tokenizer is not None:
            string = self.tokenizer(string, process_fn=self.process_fn)
        elif self.process_fn is not None:
            string = self.process_fn(string)
        return string


class LazyLoader(object):

    def __init__(self, path, data_type='data', mem_map=False, map_fn=None, is_array=False, array_data_type=np.int32,
                 load_memory=False, half_load=False):
        lazypath = get_lazy_path(path)
        datapath = os.path.join(lazypath, data_type)

        self._file = open(datapath, 'rb')
        self.file = self._file
        self.is_array = is_array
        self.array_data_type = array_data_type

        lenpath = os.path.join(lazypath, data_type + '.len.pkl')
        self.lens = pkl.load(open(lenpath, 'rb'))
        if half_load:
            self.lens = self.lens[:2 * len(self.lens) // 3]
        self.ends = list(accumulate(self.lens))
        self.dumb_ends = list(self.ends)
        self.mem_map = mem_map
        self.load_memory = load_memory
        if self.load_memory:
            data_type_size = np.dtype(self.array_data_type).itemsize
            if half_load:
                self.file = self.file.read(sum(self.lens) * data_type_size)
            else:
                self.file = self.file.read()
            self.file = np.ndarray(shape=(len(self.file) // data_type_size,), dtype=array_data_type, buffer=self.file,
                                   order='C')
        elif self.mem_map:
            if is_array:
                if self.ends[-1] == 0:
                    self.file = np.array([], dtype=array_data_type)
                else:
                    self.file = np.memmap(
                        self.file, dtype=array_data_type, mode='r', order='C')
            else:
                if self.ends[-1] == 0:
                    self.file = bytearray()
                else:
                    self.file = mmap.mmap(
                        self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.read_lock = Lock()
        self.process_fn = map_fn
        self.map_fn = map_fn
        self._tokenizer = None
        self.is_lazy = True

    def SetTokenizer(self, tokenizer):

        if tokenizer is None:
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self._tokenizer = tokenizer
        self.map_fn = ProcessorTokenizer(tokenizer, self.process_fn)

    def GetTokenizer(self):
        return self._tokenizer

    def __getitem__(self, index):

        if not isinstance(index, slice):
            if index == 0:
                start = 0
            else:
                start = self.ends[index - 1]
            end = self.ends[index]
            rtn = self.file_read(start, end)
            if self.map_fn is not None:
                rtn = self.map_fn(rtn)
        else:

            chr_lens = self.ends[index]
            if index.start == 0 or index.start is None:
                start = 0
            else:
                start = self.ends[index.start - 1]
            stop = chr_lens[-1]
            strings = self.file_read(start, stop)
            rtn = split_strings(strings, start, chr_lens)
            if self.map_fn is not None:
                rtn = [self.map_fn(s) for s in rtn]
        return rtn

    def __len__(self):
        return len(self.ends)

    def file_read(self, start=0, end=None):
        data_type_size = np.dtype(self.array_data_type).itemsize
        self.read_lock.acquire()
        if not self.mem_map and not self.load_memory:
            if self.is_array:
                start = start * data_type_size
                end = end * data_type_size if end is not None else None
            self.file.seek(start)
            if end is None:
                rtn = self.file.read()
            else:
                rtn = self.file.read(end - start)
            if self.is_array:
                rtn = np.ndarray(shape=(len(rtn) // data_type_size,),
                                 dtype=self.array_data_type, buffer=rtn, order='C')
            else:
                rtn = rtn.decode('utf-8', 'ignore')
        else:
            rtn = self.file[start:end]
            if self.is_array:
                rtn = rtn.copy()
            else:
                rtn = rtn.decode('utf-8', 'strict')
        self.read_lock.release()
        return rtn
