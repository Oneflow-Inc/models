import argparse
import sys

import oneflow as flow
from classifier_MNLI import MNLIRoBERTa
from config import infer_config
sys.path.append("../")
from tokenizer.RobertaTokenizer import RobertaTokenizer
sys.path.append("../roberta")


def inference(args):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    input_ids = tokenizer(args.text)['input_ids']
    attention_mask = tokenizer(args.text)['attention_mask']
    input_ids = flow.tensor(input_ids, dtype=flow.int32).reshape(1,-1).to(args.device)
    attention_mask = flow.tensor(attention_mask, dtype=flow.int32).reshape(1,-1).to(args.device)
    model = MNLIRoBERTa(args.pretrain_dir, args.kwargs_path,
                        args.roberta_hidden_size, args.n_classes, args.is_train).to(args.device)
    model.load_state_dict(flow.load(args.model_load_dir))
    model.eval()
    output = model(input_ids,attention_mask)
    label = flow.argmax(output)
    print(output,label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_load_dir', type=str,
                        default='./pretrain_model_MNLI')
    parser.add_argument('--kwargs_path', type=str,
                        default='./flow_roberta-base/parameters.json')
    parser.add_argument('--pretrain_dir', type=str,
                        default='./flow_roberta-base/weights')
    parser.add_argument('--text', type=str,
                        default="The new rights are nice enough	Everyone really likes the newest benefits")
    parser.add_argument('--task', type=str,
                        default='MNLI')

                
    args = parser.parse_args()
    args.device = 'cuda' if flow.cuda.is_available() else 'cpu'
    infer_config(args)
    inference(args)
