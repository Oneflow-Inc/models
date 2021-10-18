# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.append("../..")
import os
import numpy as np
import math
from tqdm import tqdm
import oneflow as flow
from classifier import MetaTeacherBERT
import tokenization
from util import Snapshot, InitNodes, Metric, CreateOptimizer, GetFunctionConfig
from data_utils.task_processors2 import processors, convert_examples_to_features, generate_dataset
from data_utils.utils import domain_list, domain_to_id, label_map
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import scipy.spatial.distance as distance
import config as configs


parser = configs.get_parser()
parser.add_argument("--task_name", type=str, default='senti', required=True, help="The name of the task to train.")
parser.add_argument("--use_domain_loss", action='store_true', help="Whether to use domain loss")
parser.add_argument('--data_portion', type=float, default=1.0, help='How many data selected.')
parser.add_argument("--domain_loss_weight", default=0.2, type=float, help="The loss weight of domain.")
parser.add_argument("--use_sample_weights",default=False, type=bool, help="The loss weight of domain.")
parser.add_argument("--vocab_file", type=str, default=None)
parser.add_argument("--train_data_prefix", type=str, default='train.of_record-')
parser.add_argument("--eval_data_prefix", type=str, default='eval.of_record-')
parser.add_argument("--dev_data_prefix", type=str, default='dev.of_record-')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--label_list", type=list, default=None)
parser.add_argument("--max_seq_length", default=128, type=int, required=False, help="The max sequence length of input sentences.")
parser.add_argument("--batch_size_per_device", default=128, type=int, required=False, help="training batch size")
parser.add_argument("--train_data_part_num", type=int, default=1, help="data part number in dataset")
parser.add_argument("--eval_data_part_num", type=int, default=1, help="data part number in dataset")
parser.add_argument("--eval_batch_size_per_device", default=128, type=int, required=False, help="evalutation batch size")
parser.add_argument("--eval_every_step_num", default=100, type=int, required=False, help="evaluate at each step")
parser.add_argument("--train_example_num", default=6480, type=int, required=False, help="The number of training data that need to calculate weighted values")
parser.add_argument("--eval_example_num", default=720, type=int, required=False, help="The number of development data that need to calculate weighted values")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument("--resave_ofrecord", action='store_true', help="Whether to resave the data to ofrecord")
parser.add_argument('--pred_distill', action='store_true')
parser.add_argument('--data_url', type=str, default="")
parser.add_argument('--temperature', type=float, default=1.)
args = parser.parse_args()

args.num_nodes = 1
args.gpu_num_per_node = 1
batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device
epoch_size = math.ceil(args.train_example_num / batch_size)
args.iter_num = epoch_size * args.num_epochs
num_eval_steps = math.ceil(args.train_example_num / eval_batch_size)


def BertDecoder(
    data_dir, batch_size, data_part_num, seq_length, part_name_prefix, shuffle=True
):
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        blob_confs = {}
        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)
        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("domain", [1])
        _blob_conf("label", [1])
        _blob_conf("idxs", [1])
        _blob_conf("weights", [1], dtype=flow.float32)
        return blob_confs

def BuildBert(
    batch_size,
    data_part_num,
    data_dir,
    part_name_prefix,
    shuffle=True
):
    hidden_size = 64 * args.num_attention_heads  # , H = 64, size per head
    intermediate_size = hidden_size * 4
    num_domains = len(domain_list[args.task_name])
    decoders = BertDecoder(
        data_dir, batch_size, data_part_num, args.seq_length, part_name_prefix, shuffle=shuffle
    )
    loss, logits = MetaTeacherBERT(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
        decoders['label'],
        decoders['domain'],
        args.vocab_size,
        input_weight=decoders['weights'],
        num_domains=num_domains,
        layer_indexes=[3, 7, 11],
        seq_length=args.seq_length,
        hidden_size=hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=args.type_vocab_size,
        initializer_range=0.02,
        get_output=False,
    )
    return loss, logits, decoders['label']

@flow.global_function(type='train', function_config=GetFunctionConfig(args))
def BertGlueFinetuneJob():
    # 跑一个batch
    loss, logits, _ = BuildBert(
        batch_size,
        args.train_data_part_num,
        os.path.join(args.data_dir, 'ofrecord', 'train'),
        args.train_data_prefix,
    )
    flow.losses.add_loss(loss)
    opt = CreateOptimizer(args)
    opt.minimize(loss)
    return {'loss': loss}


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalTrainJob():
    _, logits, label_ids = BuildBert(
        batch_size,
        args.train_data_part_num,
        os.path.join(args.data_dir, 'ofrecord', 'train'),
        args.train_data_prefix,
        shuffle=False
    )
    return logits, label_ids

@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalValJob():
    _, logits, label_ids = BuildBert(
        batch_size,
        args.eval_data_part_num,
        os.path.join(args.data_dir, 'ofrecord', 'dev'),
        args.dev_data_prefix,
        shuffle=False
    )
    return logits, label_ids


def run_eval_job(dev_job_func, num_steps, desc='dev'):
    labels = []
    predictions = []
    for index in range(num_steps):
        logits, label = dev_job_func().get()
        predictions.extend(list(logits.numpy().argmax(axis=1)))
        labels.extend(list(label))

    def metric_fn(predictions, labels):
        return {
            "accuarcy": accuracy_score(labels, predictions),
            "matthews_corrcoef": matthews_corrcoef(labels, predictions),
            "precision": precision_score(labels, predictions),
            "recall": recall_score(labels, predictions),
            "f1": f1_score(labels, predictions),
        }

    metric_dict = metric_fn(predictions, labels)
    print(desc, ', '.join('{}: {:.3f}'.format(k, v) for k, v in metric_dict.items()))
    return metric_dict['accuarcy']


def main():

    processor = processors[args.task_name]()
    args.label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)
    ofrecord_dir = os.path.join(args.data_dir, 'ofrecord')

    if not os.path.exists(ofrecord_dir) or args.resave_ofrecord:
        if args.do_train:
            train_examples = processor.get_train_examples(args.data_dir)
            print('===============================')
            print('len=', len(train_examples))
            print('===============================')
            train_feature_dict = generate_dataset(args, train_examples, tokenizer, ofrecord_dir=ofrecord_dir,
                                              stage='train', load_weight=True)
        if args.do_eval:
            dev_examples = processor.get_dev_examples(args.data_dir)
            print('===============================')
            print('len=', len(dev_examples))
            print('===============================')
            dev_feature_dict = generate_dataset(args, dev_examples, tokenizer, ofrecord_dir=ofrecord_dir,
                                                  stage='dev')

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)
    InitNodes(args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)

    print("starting meta fine-tuning")

    global_step = 0
    best_dev_acc = 0.0
    for epoch in tqdm(range(args.num_epochs)):
        metric = Metric(desc='meta-finetune', print_steps=args.loss_print_every_n_iter,
                        batch_size=batch_size, keys=['loss'])

        for step in range(epoch_size):
            global_step += 1
            loss = BertGlueFinetuneJob().async_get(metric.metric_cb(global_step, epoch=epoch))

            if global_step % args.eval_every_step_num == 0:
                print("===== evaluating ... =====")
                dev_acc = run_eval_job(
                    dev_job_func=BertGlueEvalValJob,
                    num_steps=num_eval_steps,
                    desc='dev')

                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc
                    print('===== saving model ... =====')
                    snapshot.save("best_mft_model_{}_dev_{}".format(args.task_name, best_dev_acc))

    print("best dev acc: {}".format(best_dev_acc))


if __name__ == "__main__":
    main()
