import time
import random
import pickle
import numpy as np

class DataInput:
  def __init__(self, data, batch_size, max_sl=None):

    self.batch_size = batch_size
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    #if self.epoch_size * self.batch_size < len(self.data):
    #  self.epoch_size += 1
    self.i = 0
    self.max_sl = max_sl

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    self.i += 1

    u, i, y, sl = [], [], [], []
    for t in ts:
      u.append(t[0])
      i.append(t[2])
      y.append(t[3])
      sl.append(len(t[1]))
    max_sl = max(sl)
    if self.max_sl:
        assert self.max_sl > max_sl
        max_sl = self.max_sl

    if self.max_sl:
      max_sl = self.max_sl
    hist_i = np.zeros([len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
      k += 1

    return self.i, (u, i, y, hist_i, sl)

class DataInputTest:
  def __init__(self, data, batch_size, max_sl=None):

    self.batch_size = batch_size // 2
    self.data = data
    self.epoch_size = len(self.data) // self.batch_size
    #if self.epoch_size * self.batch_size < len(self.data):
    #  self.epoch_size += 1
    self.i = 0
    self.max_sl = max_sl

  def __iter__(self):
    return self

  def __next__(self):

    if self.i == self.epoch_size:
      raise StopIteration

    ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size,
                                                  len(self.data))]
    assert len(ts) == self.batch_size
    self.i += 1

    u, i_p, i_n, y_p, y_n, sl = [], [], [], [], [], []
    for t in ts:
      u.append(t[0])
      i_p.append(t[2][0])
      i_n.append(t[2][1])
      y_p.append(1)
      y_n.append(0)
      #y.append([1])
      #y.append([0])
      sl.append(len(t[1]))
    max_sl = max(sl)
    if self.max_sl:
        assert self.max_sl > max_sl
        max_sl = self.max_sl

    hist_i = np.zeros([2 * len(ts), max_sl], np.int64)

    k = 0
    for t in ts:
      for l in range(len(t[1])):
        hist_i[k][l] = t[1][l]
        hist_i[k+self.batch_size][l] = t[1][l]
      k += 1

    return self.i, (u*2, i_p + i_n, y_p + y_n, hist_i, sl * 2)


if __name__ == "__main__":
    with open('/data/xiexuan/dataset/din_pkl/dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)
        print(user_count, item_count, cate_count)
    random.shuffle(train_set)
    batch_size = 32
    epoch_size = round(len(train_set) / batch_size)
    start = time.time()
    for i, uij in DataInputTest(test_set, batch_size, max_sl=512):
        if i % 1000 == 0:
            print("*"*30)
            for e in uij:
                n = np.array(e)
                print("test", n.shape)
            print(uij[2], uij[1])
            break
    print(time.time() - start)
    start = time.time()
    for i, uij in DataInput(train_set, batch_size):
        if i % 1000 == 0:
            for e in uij:
                n = np.array(e)
                print("test", n.shape)
            print(uij[2], uij[1])
            break
    print(time.time() - start)
    exit()

    start = time.time()
    for i, uij in DataInput(train_set, batch_size, max_sl=512):
        if i % 1000 == 0:
            print(i, np.array(uij[3]).shape)
    print(time.time() - start)
    # for python2
    # with open('/data/xiexuan/dataset/din_pkl/dataset.pkl', 'rb') as f:
    #     u = pickle._Unpickler(f)
    #     u.encoding = 'latin1'
    #     train_set = u.load()
    #     test_set = u.load()
    #     cate_list = u.load()
    #     user_count, item_count, cate_count = u.load()
    #     print(user_count, item_count, cate_count)
    #     with open('dataset.pkl', 'wb') as wf:
    #         pickle.dump(train_set, wf, pickle.HIGHEST_PROTOCOL)
    #         pickle.dump(test_set, wf, pickle.HIGHEST_PROTOCOL)
    #         pickle.dump(cate_list, wf, pickle.HIGHEST_PROTOCOL)
    #         pickle.dump((user_count, item_count, cate_count), wf, pickle.HIGHEST_PROTOCOL)