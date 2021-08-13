from transformers import GlueDataset, GlueDataTrainingArguments, RobertaTokenizer
import sys
import os
import json

data_dir = sys.argv[1]  # glue_dir
task_name = sys.argv[2]  # CoLA

glue_args = GlueDataTrainingArguments(
    task_name=task_name,
    data_dir=os.path.join(data_dir, task_name),
    max_seq_length=128,
    overwrite_cache=False
)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def convert_to_pack(item):
    pack = {
        'input_ids': item.input_ids,
        'input_mask': item.attention_mask if item.attention_mask else [1 for _ in range(glue_args.max_seq_length)],
        'segment_ids': item.token_type_ids if item.token_type_ids else [0 for _ in range(glue_args.max_seq_length)],
        'label_ids': item.label,
        'is_real_example': 1,
    } 
    return pack


def convert_to_json(dataset, split):
    fn = os.path.join( task_name, split,
                      '{}.json'.format(split))
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    result = []
    for i in range(len(dataset)):
        topack = convert_to_pack(dataset[i])
        result.append(topack)
    with open(fn, 'w') as f:
        json.dump(result,f)


def get_of_dataset():
    train_data = GlueDataset(glue_args, tokenizer, mode='train')
    eval_data = GlueDataset(glue_args, tokenizer, mode='dev')
    print('train: {}, eval: {}'.format(len(train_data), len(eval_data)))
    convert_to_json(train_data, 'train')
    convert_to_json(eval_data, 'eval')

if __name__ == '__main__':
    get_of_dataset()
