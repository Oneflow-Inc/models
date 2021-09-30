import argparse
import time

import numpy as np
import oneflow.nn as nn
import oneflow as flow
from modeling import BertForPreTraining


def _parse_args():
    parser = argparse.ArgumentParser("flags for test bert")
    parser.add_argument("--model_path", type=str, metavar="DIR", help="model path")
    parser.add_argument(
        "--use_lazy_model", type=bool, default=False, help="if loading from lazy model"
    )
    parser.add_argument("--device", type=str, default="cuda", help="model device")
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--intermediate_size",
        type=int,
        default=3072,
        help="intermediate size of bert encoder",
    )
    parser.add_argument("--max_position_embeddings", type=int, default=512)
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument("--type_vocab_size", type=int, default=2)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--hidden_size_per_head", type=int, default=64)
    parser.add_argument("--max_predictions_per_seq", type=int, default=20)

    parser.add_argument(
        "--input_path", type=str, default="", help="input string for prediction"
    )
    return parser.parse_args()


def inference(args):
    start_t = time.time()
    bert_module = BertForPreTraining(
        args.vocab_size,
        args.seq_length,
        args.hidden_size,
        args.num_hidden_layers,
        args.num_attention_heads,
        args.intermediate_size,
        nn.GELU(),
        args.hidden_dropout_prob,
        args.attention_probs_dropout_prob,
        args.max_position_embeddings,
        args.type_vocab_size,
        args.vocab_size,
    )
    end_t = time.time()
    print("Initialize model using time: {:.3f}s".format(end_t - start_t))

    start_t = time.time()
    if args.use_lazy_model:
        from utils.compare_lazy_outputs import load_params_from_lazy

        load_params_from_lazy(
            bert_module.state_dict(), args.model_path,
        )
    else:
        bert_module.load_state_dict(flow.load(args.model_path))
    end_t = time.time()
    print("Loading parameters using time: {:.3f}s".format(end_t - start_t))

    bert_module.eval()
    bert_module.to(args.device)

    class BertEvalGraph(nn.Graph):
        def __init__(self):
            super().__init__()
            self.bert = bert_module

        def build(self, input_ids, input_masks, segment_ids):
            input_ids = input_ids.to(device=args.device)
            input_masks = input_masks.to(device=args.device)
            segment_ids = segment_ids.to(device=args.device)

            with flow.no_grad():
                # 1. forward the next_sentence_prediction and masked_lm model
                _, seq_relationship_scores = self.bert(
                    input_ids, input_masks, segment_ids
                )

            return seq_relationship_scores

    bert_eval_graph = BertEvalGraph()

    start_t = time.time()
    inputs = [np.random.randint(0, 20, size=args.seq_length)]
    inputs = flow.Tensor(inputs, dtype=flow.int64, device=flow.device(args.device))
    mask = flow.cast(inputs > 0, dtype=flow.int64)

    segment_info = flow.zeros_like(inputs)
    prediction = bert_eval_graph(inputs, mask, segment_info)
    print(prediction.numpy())
    end_t = time.time()
    print("Inference using time: {:.3f}".format(end_t - start_t))


if __name__ == "__main__":
    args = _parse_args()
    inference(args)
