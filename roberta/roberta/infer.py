import argparse
import sys

import oneflow as flow
from classifier_flow import GlueRoBERTa
sys.path.append("../")
from tokenizer.RobertaTokenizer import RobertaTokenizer
sys.path.append("../roberta")


def inference(args):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    input_ids = tokenizer(args.text)['input_ids']
    attention_mask = tokenizer(args.text)['attention_mask']
    input_ids = flow.tensor(input_ids, dtype=flow.int32).reshape(1,-1).to('cuda')
    attention_mask = flow.tensor(attention_mask, dtype=flow.int32).reshape(1,-1).to('cuda')
    model = GlueRoBERTa(args.pretrain_dir, args.kwargs_path,
                        args.roberta_hidden_size, args.n_classes, args.is_train).to('cuda')
    model.load_state_dict(flow.load(args.model_load_dir))
    model.eval()
    output = model(input_ids,attention_mask)
    label = flow.argmax(output)
    # label, logits = predict(model, text)
    print(output,label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_load_dir', type=str,
                        default='./pretrain_model_SST-2')
    parser.add_argument('--kwargs_path', type=str,
                        default='./roberta_pretrain_oneflow/roberta-base/roberta-base.json')
    parser.add_argument('--pretrain_dir', type=str,
                        default='./roberta_pretrain_oneflow/roberta-base')
    parser.add_argument('--text', type=str,
                        default="this is junk food cinema at its greasiest .")

                
    args = parser.parse_args()
    args.n_classes = 2
    args.roberta_hidden_size = 768
    args.is_train = False
    inference(args)
