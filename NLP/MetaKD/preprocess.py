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
from data_utils.task_processors1 import processors, convert_examples_to_features, generate_dataset
from data_utils.utils import domain_list, domain_to_id, label_map
import scipy.spatial.distance as distance
import config as configs


parser = configs.get_parser()
parser.add_argument("--task_name", type=str, default='senti', required=True, help="The name of the task to train.")
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
parser.add_argument("--train_example_num", default=6480, type=int, required=False, help="The number of training data that need to calculate weighted values")
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument("--resave_ofrecord", action='store_true', help="Whether to resave the data to ofrecord")
args = parser.parse_args()

args.num_nodes = 1
args.gpu_num_per_node = 1
batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device
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
    output = MetaTeacherBERT(
        decoders['input_ids'],
        decoders['input_mask'],
        decoders['segment_ids'],
        decoders['label'],
        decoders['domain'],
        args.vocab_size,
        input_weight=decoders['weights'],
        num_domains=num_domains,
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
        get_output=True
    )
    return output, decoders['domain'], decoders['label'], decoders['idxs'],


@flow.global_function(type='predict', function_config=GetFunctionConfig(args))
def BertGlueEvalValJob():
    output, domains, labels, idxs = BuildBert(
        batch_size,
        args.train_data_part_num,
        os.path.join(args.data_dir, 'ofrecord', 'train'),
        args.train_data_prefix,
        shuffle=False
    )
    return output, domains, labels, idxs


def get_hidden_embedding(eval_job_func, num_steps, domain_class_embeddings, temp_output_data, desc='train'):
    labels = []
    predictions = []
    for index in tqdm(range(num_steps)):
        output, domains, labels, idxs = eval_job_func().get()
        current_size = output.shape[0]
        output = list(output.numpy().tolist())
        domains = list(domains.numpy().tolist())
        labels = list(labels.numpy().tolist())
        idxs = list(idxs.numpy().tolist())

        for i in range(current_size):
            pool_output = output[i]
            domain_name = domain_list[args.task_name][domains[i][0]]
            label = str(labels[i][0])
            domain_class_embeddings[domain_name + '\t' + label].append(pool_output)
            temp_output_data.append((idxs[i][0], domain_name, label, pool_output))

    return domain_class_embeddings, temp_output_data

def compute_weight(domain, label, current_embedding, centroid_embeddings):
    key_name = domain + "\t" + str(label)
    current_centroid = centroid_embeddings[key_name]
    other_centroids = list()
    for current_key in centroid_embeddings.keys():
        items = current_key.split("\t")
        current_domain = items[0]
        current_label = items[1]
        if not (current_domain == domain) and (current_label==label):
            other_centroids.append(centroid_embeddings[current_key])
    other_centroids = np.array(other_centroids)
    other_centroid_mean = np.mean(other_centroids, axis=0)
    first_cos_sim = 1 - distance.cosine(current_embedding, current_centroid)
    second_cos_sim = 1 - distance.cosine(current_embedding, other_centroid_mean)
    return (first_cos_sim + second_cos_sim) / 2


def main():

    processor = processors[args.task_name]()
    args.label_list = processor.get_labels()
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file)
    ofrecord_dir = os.path.join(args.data_dir, 'ofrecord')

    if not os.path.exists(ofrecord_dir) or args.resave_ofrecord:
        train_examples = processor.get_examples(args.data_dir)
        print('===============================')
        print('len=', len(train_examples))
        print('===============================')
        train_feature_dict = generate_dataset(args, train_examples, tokenizer, ofrecord_dir=ofrecord_dir,
                                              stage='train')
    domain_class_embeddings = dict()
    temp_output_data = list()

    for domain_name in domain_list[args.task_name]:
        for class_name, class_id in label_map[args.task_name].items():
            key_name = domain_name + "\t" + str(class_id)
            domain_class_embeddings[key_name] = list()

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.env.log_dir(args.log_dir)
    InitNodes(args)
    snapshot = Snapshot(args.model_save_dir, args.model_load_dir)
    domain_class_embeddings, temp_output_data = get_hidden_embedding(
        eval_job_func=BertGlueEvalValJob,
        num_steps=num_eval_steps,
        domain_class_embeddings=domain_class_embeddings,
        temp_output_data=temp_output_data,
        desc='eval')
    centroid_embeddings = dict()
    for key_name in domain_class_embeddings:
        domain_class_data_embeddings = np.array(domain_class_embeddings[key_name])
        centroid_embeddings[key_name] = np.mean(domain_class_data_embeddings, axis=0)
    records = []
    for idx, domain, label, embeddings in temp_output_data:
        weight = compute_weight(domain, label, embeddings, centroid_embeddings)
        tup = {idx: np.around(weight, decimals=5)}
        records.append(tup)

    np.save(os.path.join(ofrecord_dir, 'train/weight.npy'), records, allow_pickle=True)

if __name__ == "__main__":
    main()
