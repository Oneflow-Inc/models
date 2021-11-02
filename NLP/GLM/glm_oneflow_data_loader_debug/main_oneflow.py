import os
import time
from datetime import datetime
from arguments import get_args
from configure_data import configure_data, prepare_tokenizer

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
    prepare_dataset_end = time.time()
    print('oneflow prepare dataset time: ', prepare_dataset_end-prepare_dataset_start)
    cnt = 0
    iteration_dataset_start = time.time()
    for it in train_data:
        if cnt == 10000:
            break
        cnt += 1
    # for it in val_data:
    #     continue
    # for it in test_data:
    #     continue
    iteration_dataset_end = time.time()
    print('oneflow iteration dataset time: ', iteration_dataset_end-iteration_dataset_start)

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
