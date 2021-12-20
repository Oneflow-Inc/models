# pytorch
import torch
from torch.utils.data import Dataset, RandomSampler

import argparse

# numpy
import numpy as np
from numpy import random as ra
from collections import deque

# others
from os import path
import sys

import data_utils

#返回数据集
class CriteoDataset(Dataset):
    def __init__(
            self,
            dataset,
            max_ind_range,
            sub_sample_rate,
            randomize,
            split="train",
            raw_path="",
            pro_data="",
            memory_map=False,
            dataset_multiprocessing=False,
    ):
        '''
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing
        '''

        den_fea = 13  # 13 dense  features

        if dataset == "kaggle":
            days = 7
            out_file = "kaggleAdDisplayChallenge_processed"
        #是这个
        elif dataset == "terabyte":
            days = 1
            #days = 24
            out_file = "terabyte_processed"
        else:
            raise(ValueError("Data set option is not supported"))
        self.max_ind_range = max_ind_range
        self.memory_map = memory_map

        # split the datafile into path and filename
        #['', 'dataset', 'criteo_tb', 'day']
        lstr = raw_path.split("/")
        #数据所在的目录'/dataset/criteo_tb/'
        self.d_path = "/".join(lstr[0:-1]) + "/"
        #数据文件名'day'
        self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
        #npz文件名 '/dataset/criteo_tb/day'
        self.npzfile = self.d_path + (
            (self.d_file + "_day") if dataset == "kaggle" else self.d_file
        )
        #特征相关 '/dataset/criteo_tb/fea'
        self.trafile = self.d_path + (
            (self.d_file + "_fea") if dataset == "kaggle" else "fea"
        )

        # check if pre-processed data is available
        data_ready = True
        if memory_map:
            for i in range(days):
                reo_data = self.npzfile + "_{0}_reordered.npz".format(i)
                if not path.exists(str(reo_data)):
                    data_ready = False
        else:
            #似乎处理的数据都在一个文件
            if not path.exists(str(pro_data)):
                data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            print("Reading pre-processed data=%s" % (str(pro_data)))
            #处理后的数据，file为其路径，'../input/terabyte_processed.npz'
            file = str(pro_data)
        else:
            ## 处理原始数据，file为其路径
            print("Reading raw data=%s" % (str(raw_path)))
            file = data_utils.getCriteoAdData(
                raw_path,
                out_file,
                max_ind_range,
                sub_sample_rate,
                days,
                split,
                randomize,
                dataset == "kaggle",
                memory_map,
                dataset_multiprocessing,
            )

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # print(self.offset_per_file)

        # setup data
        if memory_map:
            # setup the training/testing split
            self.split = split
            if split == 'none' or split == 'train':
                self.day = 0
                self.max_day_range = days if split == 'none' else days - 1
            elif split == 'test' or split == 'val':
                self.day = days - 1
                num_samples = self.offset_per_file[days] - \
                              self.offset_per_file[days - 1]
                self.test_size = int(np.ceil(num_samples / 2.))
                self.val_size = num_samples - self.test_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")

            '''
            # text
            print("text")
            for i in range(days):
                fi = self.npzfile + "_{0}".format(i)
                with open(fi) as data:
                    ttt = 0; nnn = 0
                    for _j, line in enumerate(data):
                        ttt +=1
                        if np.int32(line[0]) > 0:
                            nnn +=1
                    print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                          + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # processed
            print("processed")
            for i in range(days):
                fi = self.npzfile + "_{0}_processed.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            # reordered
            print("reordered")
            for i in range(days):
                fi = self.npzfile + "_{0}_reordered.npz".format(i)
                with np.load(fi) as data:
                    yyy = data["y"]
                ttt = len(yyy)
                nnn = np.count_nonzero(yyy)
                print("day=" + str(i) + " total=" + str(ttt) + " non-zeros="
                      + str(nnn) + " ratio=" +str((nnn * 100.) / ttt) + "%")
            '''

            # load unique counts
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]
            self.m_den = den_fea  # X_int.shape[1]
            self.n_emb = len(self.counts)
            print("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

            # Load the test data
            # Only a single day is used for testing
            if self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                fi = self.npzfile + "_{0}_reordered.npz".format(
                    self.day
                )
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]          # target

        else:
            # load and preprocess data
            with np.load(file) as data:
                X_int = data["X_int"]  # continuous  feature
                X_cat = data["X_cat"]  # categorical feature
                y = data["y"]          # target
                self.counts = data["counts"]
            self.m_den = X_int.shape[1]  # den_fea
            self.n_emb = len(self.counts)
            print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

            # create reordering
            indices = np.arange(len(y))

            if split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    print("Randomized indices...")

                X_int[indices] = X_int
                X_cat[indices] = X_cat
                y[indices] = y

            else:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    print("Randomized indices per day ...")



                train_indices = np.concatenate(indices)
                test_indices = indices[-1]
                test_indices, val_indices = np.array_split(test_indices, 2)

                print("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    print("Randomized indices across days ...")

                # create training, validation, and test sets
                if split == 'train':
                    self.X_int = [X_int[i] for i in train_indices]
                    self.X_cat = [X_cat[i] for i in train_indices]
                    self.y = [y[i] for i in train_indices]
                elif split == 'val':
                    self.X_int = [X_int[i] for i in val_indices]
                    self.X_cat = [X_cat[i] for i in val_indices]
                    self.y = [y[i] for i in val_indices]
                elif split == 'test':
                    self.X_int = [X_int[i] for i in test_indices]
                    self.X_cat = [X_cat[i] for i in test_indices]
                    self.y = [y[i] for i in test_indices]

            print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        if self.memory_map:
            if self.split == 'none' or self.split == 'train':
                # check if need to swicth to next day and load data
                if index == self.offset_per_file[self.day]:
                    # print("day_boundary switch", index)
                    self.day_boundary = self.offset_per_file[self.day]
                    fi = self.npzfile + "_{0}_reordered.npz".format(
                        self.day
                    )
                    # print('Loading file: ', fi)
                    with np.load(fi) as data:
                        self.X_int = data["X_int"]  # continuous  feature
                        self.X_cat = data["X_cat"]  # categorical feature
                        self.y = data["y"]          # target
                    self.day = (self.day + 1) % self.max_day_range

                i = index - self.day_boundary
            elif self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                i = index + (0 if self.split == 'test' else self.test_size)
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")
        else:
            i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def _default_preprocess(self, X_int, X_cat, y):
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        if self.max_ind_range > 0:
            X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
        else:
            X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))

        return X_int, X_cat, y

    def __len__(self):
        if self.memory_map:
            if self.split == 'none':
                return self.offset_per_file[-1]
            elif self.split == 'train':
                return self.offset_per_file[-2]
            elif self.split == 'test':
                return self.test_size
            elif self.split == 'val':
                return self.val_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train nor test.")
        else:
            return len(self.y)
        
def collate_wrapper_criteo_offset(list_of_tuples):
    #有batch_size个
    # where each tuple is (X_int, X_cat, y)
    #第一个元素是元组，batch_size个元素，每个元素是个长度为13 的array
    #第一个元素是元组，batch_size个元素，每个元素是个长度为26 的array
    #第一个元素是元组，batch_size个元素，每个元素是个int，即label
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    #sparse特征的数量
    featureCnt = X_cat.shape[1]

    #转置了一下
    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    #每一行1-50编号，不知道为啥要这样
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    '''
    X.shape torch.Size([50, 13])
    lS_o.shape torch.Size([26, 50])
    lS_i.shape torch.Size([26, 50])
    T.shape torch.Size([50, 1])
    '''
    return X_int, torch.stack(lS_o), torch.stack(lS_i), T

 
def make_criteo_data_and_loaders(args):
    
    #得到数据集
    train_data = CriteoDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing
    )

    #collate是核对的意思
    collate_wrapper_criteo = collate_wrapper_criteo_offset

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader

# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None

if __name__ == "__main__":
    ### parse arguments ###
    parser = argparse.ArgumentParser(description="try to read data")
    parser.add_argument("--save-onnx", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mini-batch-size", type=int, default=50)
    parser.add_argument("--data-set", type=str, default="terabyte")
    parser.add_argument("--max-ind-range", type=int, default=10000000)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--raw-data-file", type=str, default="/dataset/criteo_tb/day")
    parser.add_argument("--processed-data-file", type=str, default="../input/terabyte_processed.npz")
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    args = parser.parse_args()
    train_data, train_ld=make_criteo_data_and_loaders(args)
    
    
    
    for j, inputBatch in enumerate(train_ld):
        if j == 0 and args.save_onnx:
            X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(inputBatch)
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        '''
        X.shape torch.Size([50, 13])
        lS_o.shape torch.Size([26, 50]) 不知道这个o是干嘛的
        lS_i.shape torch.Size([26, 50])
        T.shape torch.Size([50, 1])
        W.shape torch.Size([50, 1])
        '''
        print('X.shape',X.shape)
        print('lS_o.shape',lS_o.shape)
        print('lS_i.shape',lS_i.shape)
        print('T.shape',T.shape)
        print('W.shape',W.shape)