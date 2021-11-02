import os
import time
import torch
from datetime import datetime
from arguments import get_args
from configure_data import configure_data, prepare_tokenizer
from util import broadcast_data

def get_batch(data, args):
    keys = ['text', 'loss_mask']
    if args.transformer_xl or args.block_lm:
        keys += ['target', 'attention_mask']
    if args.block_lm:
        keys += ['position_id']
    datatype = torch.int64

    data_b = broadcast_data(keys, data, datatype)

    if args.transformer_xl:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].float()
        loss_mask = data_b['loss_mask'].float()
    elif args.block_lm:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].long()
        loss_mask = data_b['loss_mask'].float()
        position_ids = data_b['position_id'].long()
    else:
        tokens_ = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        labels = tokens_[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        attention_mask = None

    return tokens, labels, loss_mask, attention_mask, position_ids

def get_train_val_test_data(args, tokenizer):

    (train_data, val_data, test_data) = (None, None, None)

    data_config = configure_data()
    
    # data_set_type:"Block"
    if args.block_lm:
        data_set_type = "Block"
    elif args.transformer_xl:
        data_set_type = "GPT-XL"
    else:
        data_set_type = "GPT2"

    prepare_dataset_start = time.time()
    data_config.set_defaults(data_set_type=data_set_type, transpose=False)
    train_data, val_data, test_data = data_config.apply(args, tokenizer)
    train_data_iterator = iter(train_data)
    prepare_dataset_end = time.time()
    print('torch prepare dataset time: ', prepare_dataset_end-prepare_dataset_start)

    cnt = 0
    iteration_dataset_start = time.time()
    # for it in train_data:
    #     if cnt == 10000:
    #         break
    #     cnt += 1
    for it in range(10000):
        data = next(train_data_iterator)
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    # for it in val_data:
    #     continue
    # for it in test_data:
    #     continue
    iteration_dataset_end = time.time()
    print('torch iteration dataset time: ', iteration_dataset_end-iteration_dataset_start)

    args.do_train = 1
    args.do_valid = 1
    args.do_test = 1
    return train_data, val_data, test_data

def main():
    args = get_args()

    args.mem_length = args.mem_length if args.transformer_xl else 0

    #experiment_name:blocklm-large-blank10-11-12-02
    if args.load and not args.new_save_directory:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")
    
    #save:checkpoints/blocklm-large-blank10-11-12-02
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    
    global tokenizer
    tokenizer = prepare_tokenizer(args)
    rain_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)

if __name__ == "__main__":
    main()
