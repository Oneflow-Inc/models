# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for logging and serialization"""

import os
import random
import time
import numpy as np
import oneflow as flow
import json
import subprocess

import mpu
from tensorboardX import SummaryWriter

SUMMARY_WRITER_DIR_NAME = 'runs'


def get_log_dir(name, base):
    return os.path.join(base, SUMMARY_WRITER_DIR_NAME, name)


def get_sample_writer(log_dir, iteration=0):
    return SummaryWriter(
        log_dir=log_dir, purge_step=iteration)


def print_rank_0(message):
    # if flow.distributed.is_initialized():
    #     if flow.distributed.get_rank() == 0:
    #         print(message, flush=True)
    # else:
    print(message, flush=True)


def get_hostname():
    hostname_cmd = ["hostname -I"]
    result = subprocess.check_output(hostname_cmd, shell=True)
    master_addr = result.decode('utf-8').split()[0]
    return master_addr


def get_spare_port(args):
    if flow.distributed.get_rank() == 0:
        port = subprocess.check_output(["shuf -n 1 -i 10000-65535"], shell=True)
        port = int(port.strip())
        if port == args.master_port:
            port = subprocess.check_output(["shuf -n 1 -i 10000-65535"], shell=True)
            port = int(port.strip())
        port = flow.cuda.LongTensor([port])
    else:
        port = flow.cuda.LongTensor([0])
    flow.distributed.broadcast(port, 0)
    port = port.item()
    return port


def print_and_save_args(args, verbose=True, log_dir=None):
    if verbose:
        print('arguments:', flush=True)
        for arg in vars(args):
            dots = '.' * (29 - len(arg))
            print('  {} {} {}'.format(arg, dots, getattr(args, arg)), flush=True)
    if log_dir is not None:
        json_file = os.path.join(log_dir, "config.json")
        with open(json_file, "w") as output:
            json.dump(vars(args), output, sort_keys=True)
        if args.deepspeed and args.deepspeed_config is not None:
            with open(args.deepspeed_config) as file:
                deepspeed_config = json.load(file)
            deepspeed_json_file = os.path.join(log_dir, "config_gpt_large.json")
            with open(deepspeed_json_file, "w") as output:
                json.dump(deepspeed_config, output)


def print_params_min_max_norm(optimizer, iteration):
    index = 0
    rank = flow.distributed.get_rank()
    string = 'iteration, rank, index, model-parallel,min, max, norm\n'
    optimizer_ = optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = param.data.norm()
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.model_parallel))
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


class Timers:

    class Timer:

        def __init__(self, name):
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            assert not self.started_, 'timer has already been started'
            # flow.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            assert self.started_, 'timer is not started'
            # flow.cuda.synchronize()
            self.elapsed_ += (time.time() - self.start_time)
            self.started_ = False

        def reset(self):
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            started_ = self.started_
            if self.started_:
                self.stop()
            elapsed_ = self.elapsed_
            if reset:
                self.reset()
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time = self.timers[name].elapsed(
                reset=reset) * 1000.0 / normalizer
            string += ' | {}: {:.2f}'.format(name, elapsed_time)
        print_rank_0(string)


def report_memory(name):

    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        flow.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        flow.cuda.max_memory_allocated() / mega_bytes)
    string += ' | cached: {}'.format(flow.cuda.memory_cached() / mega_bytes)
    string += ' | max cached: {}'.format(
        flow.cuda.memory_reserved() / mega_bytes)
    print_rank_0(string)


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{}'.format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(mpu.get_model_parallel_rank()))


def ensure_directory_exists(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def get_checkpoint_tracker_filename(checkpoints_path):
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_zero_checkpoint(args, iteration, optimizer):
    zero_sd = {'iteration': iteration,
               'optimizer_state_dict': optimizer.state_dict()}
    zero_checkpoint_name = get_checkpoint_name(args.save, iteration, zero=True)
    ensure_directory_exists(zero_checkpoint_name)
    flow.save(zero_sd, zero_checkpoint_name)
    print('  successfully saved {}'.format(zero_checkpoint_name))


def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, tag=None, barrier=True,
                    only_changed_parameters=False, no_deepspeed=False, no_save_optim=False):
    
    #True
    if tag is None:
        tag = str(iteration)
    #True
    save_ds_checkpoint(iteration, model, lr_scheduler, args, tag=tag)

def save_ds_checkpoint(iteration, model, lr_scheduler, args, tag):
    sd = model.serialize(model)
    sd['iteration'] = iteration
    #True
    if lr_scheduler is not None:
        sd['client_lr_scheduler'] = lr_scheduler.state_dict()
    #True
    if not args.no_save_rng:
        sd['random_rng_state'] = random.getstate()
        sd['np_rng_state'] = np.random.get_state()

    flow.save(model.state_dict(), os.path.join(args.save, str(iteration)+"_glm_model.pt"))
    
    # model.save_checkpoint(args.save, tag, client_state=sd)


def get_checkpoint_iteration(load_path):
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank_0('WARNING: could not find the metadata file {} '.format(
            tracker_filename))
        if os.path.isdir(load_path):
            path = os.path.normpath(load_path)
            load_dir, tag = os.path.split(path)
            print_rank_0('Try to directly load the checkpoint from the directory')
            return load_dir, tag, False, True
        print_rank_0('    will not load any checkpoints and will start from '
                     'random')
        return load_path, 0, False, False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        release = metastring == 'release'
      
    return load_path, metastring, release, True


def load_checkpoint(model, optimizer, lr_scheduler, args, no_deepspeed=False, no_load_optim=False):
    load_dir, tag, release, success = get_checkpoint_iteration(args.load)

    if not success:
        return 0

    if args.deepspeed and not no_deepspeed:

        checkpoint_name, sd = model.load_checkpoint(load_dir, tag,
                                                    load_optimizer_states=not args.no_load_optim and not no_load_optim,
                                                    load_lr_scheduler_states=not args.no_load_lr_scheduler)
        if not args.no_load_lr_scheduler and "client_lr_scheduler" in sd:
            lr_scheduler.load_state_dict(sd["client_lr_scheduler"])
            print_rank_0("Load lr scheduler state")
        if checkpoint_name is None:
            if mpu.get_data_parallel_rank() == 0:
                print("Unable to load checkpoint.")
            return tag

    else:

        checkpoint_name = get_checkpoint_name(load_dir, tag, release)

        if mpu.get_data_parallel_rank() == 0:
            print('global rank {} is loading checkpoint {}'.format(
                flow.distributed.get_rank(), checkpoint_name))

        sd = flow.load(checkpoint_name, map_location='cpu')

        if args.deepspeed:
            model = model.module
        missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)
        if missing_keys or unexpected_keys:
            print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")

        if not release and not args.finetune and not args.no_load_optim and not no_load_optim:
            try:
                if optimizer is not None:
                    optimizer.load_state_dict(sd['optimizer'])
                if lr_scheduler is not None:
                    lr_scheduler.load_state_dict(sd['lr_scheduler'])
            except KeyError:
                print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                             'Specify --no-load-optim or --finetune to prevent '
                             'attempting to load the optimizer '
                             'state.'.format(checkpoint_name))

    if args.finetune or release:
        iteration = 0
    else:
        try:
            iteration = sd['iteration']
        except KeyError:
            try:  
                iteration = sd['total_iters']
            except KeyError:
                print_rank_0('A metadata file exists but Unable to load iteration '
                             ' from checkpoint {}, starting from 0 iteration'.format(checkpoint_name))
                iteration = 0

    return iteration


def load_weights(src, dst, dst2src=False):
    conv_layer = 'Conv1D' in str(type(src))
    for n, p in src.named_parameters():
        if dst2src:
            data = dst._parameters[n].data
            load = p.data
        else:
            data = p.data
            load = dst._parameters[n].data
        if conv_layer and 'weight' in n:
            data = data.t().contiguous()
        load.copy_(data)


def load_mlp(our, oai, dst2src=False):
    load_weights(oai.c_fc, our.dense_h_to_4h, dst2src)
    load_weights(oai.c_proj, our.dense_4h_to_h, dst2src)


def load_attention(our, oai, dst2src=False):
    load_weights(oai.c_attn, our.query_key_value, dst2src)
    load_weights(oai.c_proj, our.dense, dst2src)


def load_transformer_layer(our, oai, dst2src=False):
    load_weights(oai.ln_1, our.input_layernorm, dst2src)
    load_weights(oai.ln_2, our.post_attention_layernorm, dst2src)
    load_mlp(our.mlp, oai.mlp, dst2src)
    load_attention(our.attention, oai.attn, dst2src)


def move_weights(our, oai, dst2src=False):
    transformer_model = oai.transformer
    load_weights(transformer_model.ln_f, our.transformer.final_layernorm, dst2src)
    load_weights(transformer_model.wte, our.word_embeddings, dst2src)
    load_weights(transformer_model.wpe, our.position_embeddings, dst2src)

    for our_layer, oai_layer in zip(our.transformer.layers, oai.transformer.h):
        load_transformer_layer(our_layer, oai_layer, dst2src)


def debug_finetune_data(local_vars, batch_id, tokenizer):
    tokens, target_ids = local_vars["tokens"], local_vars["target_ids"]
    attention_mask, logit_mask, position_ids = local_vars["attention_mask"], local_vars["logit_mask"], local_vars[
        "position_ids"]
    output_tokens = []
    sep = attention_mask[batch_id].item()
    for i, token in enumerate(tokens[batch_id][:sep].tolist()):
        token = tokenizer.IdToToken(token)
        if token == '[MASK]':
            token = f"[{position_ids[batch_id][0, i].item()}]"
        output_tokens.append(token)
    print(" ".join(output_tokens))
    target_positions = []
    for i in range(sep, tokens.size(-1)):
        if logit_mask[batch_id][i]:
            target_positions.append(i)
    print(target_positions)
    print(tokenizer.DecodeIds(tokens[batch_id][target_positions].tolist()))
    if len(target_ids.shape) > 2:
        print(tokenizer.DecodeIds(target_ids[batch_id][target_positions].tolist()))
    else:
        print(tokenizer.DecodeIds(target_ids[batch_id].tolist()))
    print(position_ids[batch_id][:, target_positions])



_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    for key in keys:
        assert data[key].dtype == target_dtype, '{} has data type {} which '\
            'is different than {}'.format(key, data[key].dtype, target_dtype)


def _build_key_size_numel_dictionaries(keys, data):
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]
    
    # if get_model_parallel_rank() == 0:
    offset = 0
    for key in keys:
        assert data[key].dim() < max_dim, 'you should increase MAX_DATA_DIM'
        size = data[key].size()
        for i, s in enumerate(size):
            sizes[i + offset] = s
        offset += max_dim

    # sizes_cuda = torch.cuda.LongTensor(sizes)
    # torch.distributed.broadcast(sizes_cuda, get_model_parallel_src_rank(),
    #                             group=get_model_parallel_group())
    
    sizes_cpu = flow.Tensor(sizes).to(flow.int64)
    # sizes_cpu = sizes_cuda.cpu()
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys,
                                                                          data)

    # if get_model_parallel_rank() == 0:
    _check_data_types(keys, data, datatype)
    
    flatten_data = flow.cat(
            [data[key].contiguous().view((-1,)) for key in keys], dim=0).cuda()
    # else:
    #     flatten_data = torch.empty(total_numel,
    #                                device=torch.cuda.current_device(),
    #                                dtype=datatype)

    # torch.distributed.broadcast(flatten_data, get_model_parallel_src_rank(),
    #                             group=get_model_parallel_group())

    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        for i in range(len(size)):
            size[i] = int(size[i].numpy())
        numel = int(key_numel[key].numpy())
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel
    return output

class VocabUtility:

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)

def get_loss(vocab_parallel_logits,target):
    logits = vocab_parallel_logits.clone()
       
    logits_max = flow.max(logits, dim=-1)[0]
    
    # flow.distributed.all_reduce(logits_max,
    #                                 op=flow.distributed.ReduceOp.MAX,
    #                                 group=get_model_parallel_group())
    
    # logits.sub_(logits_max.unsqueeze(dim=-1))
    logits = logits - logits_max.unsqueeze(dim=-1)

    exp_logits = logits.exp()

    sum_exp_logits = exp_logits.sum(dim=-1)
    
    # flow.distributed.all_reduce(sum_exp_logits,
    #                                 op=flow.distributed.ReduceOp.SUM,
    #                                 group=get_model_parallel_group())

    get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    # rank = get_model_parallel_rank()
    # world_size = get_model_parallel_world_size()
    rank = 0
    world_size = 1
    vocab_start_index, vocab_end_index = get_vocab_range(
        partition_vocab_size, rank, world_size)
    
    target_mask = ((target < vocab_start_index) | (target >= vocab_end_index))
    masked_target = (target.clone() - vocab_start_index)
   
    # del
    # masked_target[target_mask] = 0
    
    logits_2d = logits.view(-1, partition_vocab_size)

    masked_target_1d = masked_target.view((-1,)).to(flow.int)

    arange_1d = flow._C.arange(start=0, end=logits_2d.size()[0],
                                device=logits_2d.device).to(flow.int)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits = predicted_logits_1d.reshape(*target.size())
    # del
    # predicted_logits[target_mask] = 0.0

    loss = flow.log(sum_exp_logits) - predicted_logits
    return loss
