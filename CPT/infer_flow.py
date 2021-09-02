import argparse

import oneflow as flow

from classifier_flow import ClueAFQMCCPT
from tokenizer.tokenization_bert import BertTokenizer


def inference_afqmc(args):

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = ClueAFQMCCPT(args.pretrain_dir, args.n_classes,
                         args.is_train).to(args.device)

    vec = tokenizer(args.text1, args.text2)
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    input_ids = flow.tensor(input_ids, dtype=flow.int32).reshape(
        1, -1).to(args.device)
    attention_mask = flow.tensor(
        attention_mask, dtype=flow.int32).reshape(1, -1).to(args.device)

    model.load_state_dict(flow.load(args.model_load_dir))
    model.eval()
    output = model(input_ids, attention_mask)
    output = flow.softmax(output)
    label = flow.argmax(output)
    print("Softmax output:", output.numpy())
    print("Predict:", label.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_dir', type=str,
                        default='/remote-home/share/shxing/cpt_pretrain_oneflow/cpt-base')
    parser.add_argument('--model_load_dir', type=str,
                        default='cpt_pretrain_afqmc')
    parser.add_argument('--text1', type=str, default="双十一花呗提额在哪")
    parser.add_argument('--text2', type=str, default="里可以提花呗额度")
    parser.add_argument('--task', type=str, default="afqmc")
    parser.add_argument('--cuda', action="store_true")

    args = parser.parse_args()
    args.is_train = False
    args.device = "cuda" if args.cuda else "cpu"

    if args.task == "afqmc":
        args.n_classes = 2
        inference_afqmc(args)
    else:
        raise NotImplementedError


