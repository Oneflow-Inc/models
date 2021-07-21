import numpy as np
from glob import glob
import spacy
import re
from tqdm import tqdm
spacy_en = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'ner', 'textcat'
                                    'entity_ruler', 'sentencizer', 
                                    'merge_noun_chunks', 'merge_entities',
                                    'merge_subtokens'])
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_dataset(path):
    data_set = []
    data_labels = []
    all_neg_f = glob(path+'/neg/*.txt')
    for f in tqdm(all_neg_f):
        data_labels.append(0)
        with open(f) as neg_sample:
            for line in neg_sample:
                line = clean_str(line)
                line = tokenizer(line)
                data_set.append(line)
    all_pos_f = glob(path+'/pos/*.txt')
    for f in tqdm(all_pos_f):
        data_labels.append(1)
        with open(f) as pos_sample:
            for line in pos_sample:
                line = clean_str(line)
                line = tokenizer(line)
                data_set.append(line)
    
    return data_set, data_labels
    
def build_vocab_from_pretrained_embed():
    word_to_idx = {}
    word_embeddings = []
    word_to_idx["<UNK>"] = 0
    word_to_idx["<PAD>"] = 1 
    word_embeddings.append([0.] * 100)
    word_embeddings.append([0.] * 100)
    with open('./glove.6B.100d.txt') as file:
        global_idx = 2
        for line in tqdm(file):
            tmp = []
            for idx,x in enumerate(line.strip().split(' ')):
                if idx == 0:
                    word_to_idx[x] = global_idx
                    global_idx += 1
                else:
                    tmp.append(float(x))
            word_embeddings.append(tmp)
    return word_to_idx,np.array(word_embeddings)

def build_vocab(dataset):
    word_to_idx = {}
    word_to_idx["<UNK>"] = 0
    word_to_idx["<PAD>"] = 1
    global_idx = 2
    for line in tqdm(dataset):
        tmp = []
        for x in line:
            if x not in word_to_idx:
                word_to_idx[x] = global_idx
                global_idx += 1
    return word_to_idx

def tensorize_data(data,
                   vocab_dct,
                   max_len = 200):
    tensorized_data = []
    for line in data:     
        tmp = []
        count = 0
        for x in line:
            if count < max_len:
                if x in vocab_dct:
                    tmp.append(vocab_dct[x])
                else:
                    tmp.append(vocab_dct['<UNK>'])
                count += 1
        if count < max_len:
            tmp += [1] *(max_len - count)
        tensorized_data.append(tmp)
    return tensorized_data