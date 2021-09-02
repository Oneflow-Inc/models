"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import math
import argparse
from datetime import datetime

import oneflow as flow
import oneflow.nn as nn
import config
from config import str2bool
from compare_lazy_outputs import load_params_from_lazy

from squad import SQuAD
from squad_util import RawResult, gen_eval_predict_json
from utils.optimizer import build_adamW_optimizer
from utils.lr_scheduler import PolynomialLR

parser = config.get_parser()
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument("--train_data_dir", type=str, default=None)
parser.add_argument("--train_example_num", type=int, default=88614, 
                    help="example number in dataset")
parser.add_argument("--batch_size_per_device", type=int, default=32)
parser.add_argument("--train_data_part_num", type=int, default=1, 
                    help="data part number in dataset")
parser.add_argument("--eval_data_dir", type=str, default=None)
parser.add_argument("--eval_example_num", type=int, default=10833, 
                    help="example number in dataset")
parser.add_argument("--eval_batch_size_per_device", type=int, default=64)
parser.add_argument("--eval_data_part_num", type=int, default=1, 
                    help="data part number in dataset")
parser.add_argument("--device", type=str, default="cuda", help="training device")

# post eval
parser.add_argument("--output_dir", type=str, default='squad_output', help='folder for output file')
parser.add_argument("--doc_stride", type=int, default=128)
parser.add_argument("--max_seq_length", type=int, default=384)
parser.add_argument("--max_query_length", type=int, default=64)
parser.add_argument("--vocab_file", type=str,
                    help="The vocabulary file that the BERT model was trained on.")
parser.add_argument("--predict_file", type=str, 
                    help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
parser.add_argument("--n_best_size", type=int, default=20,
    help="The total number of n-best predictions to generate in the nbest_predictions.json output file.")
parser.add_argument("--max_answer_length", type=int, default=30,
    help="The maximum length of an answer that can be generated. This is needed \
    because the start and end predictions are not conditioned on one another.")
parser.add_argument("--verbose_logging", type=str2bool, default='False',
    help="If true, all of the warnings related to data processing will be printed. \
    A number of warnings are expected for a normal SQuAD evaluation.")
parser.add_argument("--version_2_with_negative", type=str2bool, default='False',
    help="If true, the SQuAD examples contain some that do not have an answer.")
parser.add_argument("--null_score_diff_threshold", type=float, default=0.0,
    help="If null_score - best_non_null is greater than the threshold predict null.")

args = parser.parse_args()

batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device
device = flow.device(args.device)

epoch_size = math.ceil(args.train_example_num / batch_size)
num_eval_steps = math.ceil(args.eval_example_num / eval_batch_size)
args.iter_num = epoch_size * args.num_epochs
args.predict_batch_size = eval_batch_size
config.print_args(args)

def save_model(module: nn.Module, checkpoint_path: str, name: str):
    snapshot_save_path = os.path.join(checkpoint_path, f"snapshot_{name}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print(f"Saving model to {snapshot_save_path}")
    flow.save(
        module.state_dict(),
        snapshot_save_path)


class SquadDecoder(nn.Module):
    def __init__(self, data_dir, batch_size, data_part_num, seq_length, is_train=True):
        super().__init__()
        self.is_train = is_train

        ofrecord = nn.OFRecordReader(data_dir, batch_size=batch_size, data_part_num=data_part_num, random_shuffle=is_train, shuffle_after_epoch=is_train)
        self.ofrecord = ofrecord

        blob_confs = {}
        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = nn.OFRecordRawDecoder(name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        if is_train:
            _blob_conf("start_positions", [1])
            _blob_conf("end_positions", [1])
        else:
            _blob_conf("unique_ids", [1])

        self.blob_confs = blob_confs
    
    def forward(self):
        data_record = self.ofrecord()
        input_ids = self.blob_confs["input_ids"](data_record)
        input_mask = self.blob_confs["input_mask"](data_record)
        segment_ids = self.blob_confs["segment_ids"](data_record)
        if self.is_train:
            start_positions = self.blob_confs["start_positions"](data_record)
            end_positions = self.blob_confs["end_positions"](data_record)
            return (input_ids, input_mask, segment_ids, start_positions, end_positions)
        else:
            unique_ids = self.blob_confs["unique_ids"](data_record)
            return (input_ids, input_mask, segment_ids, unique_ids)
            
    
def squad_finetune(epoch:int, iter_per_epoch: int, model:nn.Graph, print_steps: int):

    total_loss = []
    for i in range(iter_per_epoch):

        loss = model()
        total_loss.append(loss.numpy().item())

        if (i + 1) % print_steps == 0:
            print(f"{epoch}/{i+1}, total loss: {total_loss[-10:]/min(10, len(total_loss)):.6f}")

def squad_eval(num_eval_steps:int, model: nn.Graph, print_steps: int):
    all_results = []
    for step in range(num_eval_steps):
        unique_ids, start_logits, end_logits = model()
        unique_ids = unique_ids.numpy()
        start_logits = start_logits.numpy()
        end_logits = end_logits.numpy()

        for unique_id, start_logit, end_logit in zip(unique_ids, start_logits, end_logits):
            all_results.append(RawResult(
                    unique_id = int(unique_id[0]),
                    start_logits = start_logit.flatten().tolist(),
                    end_logits = end_logit.flatten().tolist(),
                ))
        
        if step % print_steps == 0:
            print("{}/{}, num of results:{}".format(step, num_eval_steps, len(all_results)))
            print("last uid:", unique_id[0])

    gen_eval_predict_json(args, all_results)
    

def main():

    hidden_size = 64 * args.num_attention_heads  # H = 64, size per head
    intermediate_size = hidden_size * 4

    print("Create Bert model for SQuAD")
    squad_model = SQuAD(args.vocab_size,
            seq_length=args.seq_length,
            hidden_size=hidden_size,
            hidden_layers=args.num_hidden_layers,
            atten_heads=args.num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=nn.GELU(),
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            max_position_embeddings=args.max_position_embeddings,
            type_vocab_size=args.type_vocab_size,
            initializer_range=0.02)
    
    # Load pretrain model from lazy trained model
    load_params_from_lazy(squad_model.state_dict(), args.model_load_dir)

    squad_model.to(device)

    if args.do_train:
        print("Create SQuAD training data decoders")
        test_decoders = SquadDecoder(args.train_data_dir, batch_size, args.train_data_part_num, args.seq_length)

        optimizer = build_adamW_optimizer(squad_model, args.learning_rate, args.weight_decay_rate, weight_decay_excludes=["bias", "LayerNorm", "layer_norm"])

        lr_scheduler = PolynomialLR(
                        optimizer, 
                        steps=args.iter_num, 
                        end_learning_rate=0.0)

        warmup_batches = int(args.iter_num * args.warmup_proportion)
        lr_scheduler = flow.optim.lr_scheduler.WarmUpLR(
                        lr_scheduler, 
                        warmup_factor=0, 
                        warmup_iters=warmup_batches, 
                        warmup_method="linear")

        class SQuADGraph(nn.Graph):
            def __init__(self):
                super().__init__()
                self.squad_model = squad_model
                self.criterion = nn.CrossEntropyLoss()

                self.add_optimizer(optimizer, lr_sch=lr_scheduler)
                self._decoders = test_decoders
            
            def build(self):
                (input_ids, input_mask, segment_ids, start_positions, end_positions) = self._decoders()
                input_ids = input_ids.to(device=device)
                input_mask = input_mask.to(device=device)
                segment_ids = segment_ids.to(device=device)
                start_positions = start_positions.to(device=device)
                end_positions = end_positions.to(device=device)

                start_logits, end_logits = self.squad_model(input_ids, segment_ids, input_mask)
                start_logits = flow.reshape(start_logits, [-1, args.seq_length])
                end_logits = flow.reshape(end_logits, [-1, args.seq_length])

                start_loss = self.criterion(start_logits, start_positions)
                end_loss = self.criterion(end_logits, end_positions)
                total_loss = (start_loss + end_loss) * 0.5
                total_loss.backward()

                return total_loss

        squad_graph = SQuADGraph()

        for epoch in range(args.num_epochs):
            squad_model.train()
            squad_finetune(epoch, epoch_size, squad_graph, args.loss_print_every_n_iter)

        if args.save_last_snapshot:
            save_model(squad_model, args.model_save_dir, "last_snapshot")   
            

    if args.do_eval:
        assert os.path.isdir(args.eval_data_dir)
        print("Create SQuAD testing data decoders")
        test_decoders = SquadDecoder(args.eval_data_dir, eval_batch_size, args.eval_dat_part_num, args.seq_length, is_train=False)

        class SQuADEvalGraph(nn.Graph):
            def __init__(self):
                super().__init__()
                self.squad_model = squad_model
                self._decoders = test_decoders
            
            def build(self):    
                (input_ids, input_mask, segment_ids, unique_ids) = self._decoders()
                input_ids = input_ids.to(device=device)
                input_mask = input_mask.to(device=device)
                segment_ids = segment_ids.to(device=device)
                unique_ids = unique_ids.to(device=device)

                with flow.no_grad():
                    start_logits, end_logits = self.squad_model(input_ids, segment_ids, input_mask)

                return unique_ids, start_logits, end_logits

        squad_eval_graph = SQuADEvalGraph()

        squad_eval(num_eval_steps, squad_eval_graph, args.loss_print_every_n_iter)
        
        
        
if __name__ == "__main__":
    main()
