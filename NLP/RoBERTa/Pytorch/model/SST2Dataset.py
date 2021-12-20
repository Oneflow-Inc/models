import os
import json

import torch 
from torch.utils.data import Dataset

task_name = 'SST-2'  # CoLA or SST-2

def read_data(split):
    fn = os.path.join(task_name, split,
                      '{}.json'.format(split))
    input_ids = []
    attention_mask = []
    labels = []
    with open(fn, 'r') as f:
        result = json.load(f)
        for pack_data in result:
            input_ids.append(pack_data["input_ids"])
            # input_mask是什么？是说这个句子多长吗？
            attention_mask.append(pack_data["input_mask"])
            labels.append(pack_data["label_ids"])
    input_ids = torch.tensor(input_ids, dtype=torch.int32)
    attention_mask = torch.tensor(attention_mask, dtype=torch.int32)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels


class SST2Dataset(Dataset):
  def __init__(self, input_ids, attention_mask, labels):
    super(SST2Dataset, self).__init__()
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.labels = labels

  def __getitem__(self, key):
    return self.input_ids[key], self.attention_mask[key], self.labels[key]

  def __len__(self):
    return self.input_ids.shape[0]


