
import os
import tqdm
import random
import pickle

import numpy as np
import torch

class GPTDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        file_path, 
        tokenizer, 
        block_size: int,
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        directory, filename = os.path.split(file_path)
        cached_file = os.path.join(directory, f'cached_{block_size}_{filename}')
        if os.path.exists(cached_file):
            print("loading features from cached file")
            with open(cached_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            print("creating features from dataset file")
            self.examples = []
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            tokenized_text = self.tokenizer.tokenize(text)
            while len(tokenized_text) >= self.block_size:
                self.examples.append(tokenized_text[:block_size])
                tokenized_text = tokenized_text[block_size:]

            print("saving features into cached file")
            with open(cached_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return torch.tensor(example, dtype=torch.long)
