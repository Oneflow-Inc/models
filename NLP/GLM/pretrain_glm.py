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

"""Pretrain GPT2"""

from datetime import datetime
import os
import random
import math

import oneflow.distributed
from filelock import FileLock
import numpy as np
import oneflow as flow

import deepspeed
from contextlib import ExitStack
from arguments import get_args
from configure_data import configure_data, prepare_tokenizer, build_multi_task_dataset
import pathlib

from train_utils import setup_model_and_optimizer, train_step
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_and_save_args
from utils import print_rank_0
from utils import get_sample_writer, get_log_dir, get_hostname
from utils import broadcast_data, get_loss
import oneflow.distributed as dist


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               set_loss_mask=False,
                               mem_length=None):
    batch_size, seq_length = data.size()

    if mem_length:
        if attention_mask is None:
            attention_mask = flow.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = flow.tril(flow.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = flow.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = flow.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    if loss_mask is None:
        loss_mask = flow.ones(data.size(), dtype=flow.float, device=data.device)

    position_ids = flow._C.arange(seq_length, dtype=flow.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if set_loss_mask:
        loss_mask[data == eod_token] = 0.0
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):

            eod_index = position_ids[b, data[b] == eod_token]
            if reset_position_ids:
                eod_index = eod_index.clone()

            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
             
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
             
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids




def get_batch(data, args):
    keys = ['text', 'loss_mask']
    if args.transformer_xl or args.block_lm:
        keys += ['target', 'attention_mask']
    if args.block_lm:
        keys += ['position_id']
    datatype = flow.int64

    data_b = broadcast_data(keys, data, datatype)

    if args.transformer_xl:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].float()
        loss_mask = data_b['loss_mask'].float()
    elif args.block_lm:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].long()
        loss_mask = data_b['loss_mask'].float()
        position_ids = data_b['position_id'].long()
    else:
        tokens_ = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        labels = tokens_[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        attention_mask = None

 
    if not args.block_lm:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            args.reset_position_ids,
            args.reset_attention_mask,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            mem_length=args.mem_length,
            set_loss_mask=not args.transformer_xl)

        if args.fp16:
            attention_mask = attention_mask.half()
    return tokens, labels, loss_mask, attention_mask, position_ids


tokenizer = None


def forward_step(data_iterator, model, args, timers, mems):
  
    timers('batch generator').start()
    timers('data loader').start()
    
    rand = random.Random(args.iteration * 1 + 0)

    if data_iterator[1] and rand.random() < args.multi_task_ratio:
        data = next(data_iterator[1]) if data_iterator[1] else None
        data["mode"] = "multi-task"
    else:
        data = next(data_iterator[0]) if data_iterator[0] else None
    
    timers('data loader').stop()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    timers('batch generator').stop()
    
    def print_masked_text(batch_id):
        block_position_ids = position_ids[:, 1]
        position_ids_ = position_ids[:, 0]
        sep = attention_mask.item() if flow.numel(attention_mask) == 1 else attention_mask[batch_id].item()
        text, last_segment = "", []
        for i, token_id in enumerate(tokens[batch_id, :sep].tolist()):
            token = tokenizer.IdToToken(token_id)
            if token.startswith('[MASK') or token.endswith('MASK]'):
                if last_segment:
                    text += tokenizer.DecodeIds(last_segment)
                    last_segment = []
                text += f" [{position_ids_[batch_id, i].item()}, {token}]"
            else:
                last_segment.append(token_id)
        if last_segment:
            text += tokenizer.DecodeIds(last_segment)
        print(text.encode('utf-8'))
        last_index = None
        for i in range(sep, tokens.size(1)):
            if tokenizer.IdToToken(tokens[batch_id, i].item()).startswith("<|startofpiece"):
                if last_index is not None:
                    print(tokenizer.DecodeIds(tokens[batch_id, last_index: i].tolist()).encode('utf-8'), "|",
                          tokenizer.DecodeIds(labels[batch_id, last_index: i].tolist()).encode('utf-8'),
                          position_ids_[batch_id, last_index: i].tolist(),
                          block_position_ids[batch_id, last_index:i].tolist())
                last_index = i
        if last_index is not None:
            print(tokenizer.DecodeIds(tokens[batch_id, last_index:].tolist()).encode('utf-8'), "|",
                  tokenizer.DecodeIds(labels[batch_id, last_index:].tolist()).encode('utf-8'),
                  position_ids_[batch_id, last_index:].tolist(), block_position_ids[batch_id, last_index:].tolist())

    if data is not None and "mode" in data:
        mode = data['mode']
    else:
        mode = 'bert'
    
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    
    losses = get_loss(logits.contiguous().float(),labels)
    
    loss_mask = loss_mask.view((-1,))
    loss = flow.sum(losses.view((-1,)) * loss_mask)
    
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    with open("loss.txt",'w') as f:
        f.write(str(loss.item())+'\n')
    # print(loss)
    return loss, mems, mode


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    # if args.fp16:
    #     log_string += ' loss scale {:.1f} |'.format(
    #        optimizer.cur_scale if args.deepspeed else  optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step):
    string = ' validation loss at {}'.format(prefix)
    string += ' | LM loss: {:.6E}'.format(loss)
    string += ' | LM PPL: {:.6E}'.format(ppl)
    if gpt_loss != 0:
        string += ' | GPT loss: {:.6E}'.format(gpt_loss)
    if bert_loss != 0:
        string += ' | BERT loss: {:.6E}'.format(bert_loss)
    if sent_loss != 0:
        string += ' | Sent loss: {:.6E}'.format(sent_loss)
    if multi_loss != 0:
        string += ' | Multi loss: {:.6E}'.format(multi_loss)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        if gpt_loss != 0:
            summary_writer.add_scalar(f'Train/valid_gpt_loss', gpt_loss, step)
        if bert_loss != 0:
            summary_writer.add_scalar(f'Train/valid_bert_loss', bert_loss, step)
        if sent_loss != 0:
            summary_writer.add_scalar(f'Train/valid_sent_loss', sent_loss, step)
        if multi_loss != 0:
            summary_writer.add_scalar(f'Train/valid_multi_loss', multi_loss, step)


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    model.train()

    total_lm_loss = 0.0

    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []
    
    import time
    tb = time.time()
    #0,200000
    while args.iteration < 1000:
    # while args.iteration < args.train_iters:
        lm_loss, skipped_iter, mems = train_step(train_data_iterator,
                                                 model,
                                                 optimizer,
                                                 lr_scheduler,
                                                 args, timers, mems=mems, forward_step_func=forward_step)
        skipped_iters += skipped_iter
        args.iteration += 1
        # print(args.iteration)
        total_lm_loss += lm_loss.data.detach().float()
       
        #True
        if False:
        # if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
        
        if False:
        # if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)
        
        if False:
        # if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, verbose=False, step=args.iteration,
                summary_writer=summary_writer, forward_step_func=forward_step)
    te = time.time()
    print(te-tb)
    exit(0)
    return args.iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, forward_step_func, verbose=False):

    model.eval()

    total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss = 0, 0, 0, 0, 0
    gpt_iters, bert_iters, sent_iters, multi_iters = 0, 0, 0, 0
    mems = []
    with flow.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            
            lm_loss, mems, mode = forward_step_func(data_iterator, model, args, timers, mems=mems)
            
            # if args.deepspeed and args.deepspeed_activation_checkpointing:
            #     deepspeed.checkpointing.reset()

            lm_loss = lm_loss.data.detach().float().item()
            total_lm_loss += lm_loss
            if mode == 'gpt':
                total_gpt_loss += lm_loss
                gpt_iters += 1
            elif mode == 'bert':
                total_bert_loss += lm_loss
                bert_iters += 1
            elif mode == 'sentence':
                total_sent_loss += lm_loss
                sent_iters += 1
            elif mode == 'multi-task':
                total_multi_loss += lm_loss
                multi_iters += 1

    model.train()
    
    loss_data = flow.Tensor(
        [total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss, gpt_iters, bert_iters,
         sent_iters, multi_iters]).cuda()
    
    loss_data = loss_data.tolist()
    total_lm_loss = loss_data[0] / args.eval_iters / (args.world_size / args.model_parallel_size)
    total_gpt_loss = loss_data[1] / loss_data[5] if loss_data[5] > 0 else 0
    total_bert_loss = loss_data[2] / loss_data[6] if loss_data[6] > 0 else 0
    total_sent_loss = loss_data[3] / loss_data[7] if loss_data[7] > 0 else 0
    total_multi_loss = loss_data[4] / loss_data[8] if loss_data[8] > 0 else 0

    return total_lm_loss, total_gpt_loss, total_bert_loss, total_sent_loss, total_multi_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, forward_step_func, verbose=False, step=None, summary_writer=None):
    lm_loss, gpt_loss, bert_loss, sent_loss, multi_loss = evaluate(data_iterator, model, args, timers, verbose=verbose,
                                                                   forward_step_func=forward_step_func)

    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, gpt_loss, bert_loss, sent_loss, multi_loss, step)

    return lm_loss


def set_random_seed(seed):

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        flow.manual_seed(seed)


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

    data_config.set_defaults(data_set_type=data_set_type, transpose=False)
    
    train_data, val_data, test_data = data_config.apply(args, tokenizer)

    args.do_train = 1
    args.do_valid = 1
    args.do_test = 1
    return train_data, val_data, test_data


def main():
    
    flow.backends.cudnn.enabled = False

    timers = Timers()
    
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
   
    # initialize_distributed(args)
    
    # set_random_seed(args.seed)

    global tokenizer
    tokenizer = prepare_tokenizer(args)

    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)

    multi_train_data, multi_val_data = None, None
    
    #False
    if args.multi_task_ratio > 0.0:
        multi_train_data, multi_val_data = build_multi_task_dataset(args, tokenizer)

    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    
    #False
    if args.load is not None:
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    #False
    if args.switch_linear:
        lr_scheduler.switch_linear(args)

    summary_writer = None
    # if flow.distributed.get_rank() == 0:
    print('Pretrain GPT2 model')
    args.log_dir = None
    if args.train_iters > 0:
        args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
        summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
    print_and_save_args(args, verbose=True, log_dir=args.log_dir)
    
    #True
    if args.resume_dataloader:
        print_rank_0("Resume dataloader")
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % len(train_data)
        if val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters
            val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
        if multi_train_data is not None:
            multi_train_data.batch_sampler.start_iter = int(args.iteration * args.multi_task_ratio) % len(
                multi_train_data)
        if multi_val_data is not None:
            start_iter_val = (args.iteration // args.eval_interval) * args.eval_iters * args.multi_task_ratio
            multi_val_data.batch_sampler.start_iter = start_iter_val % len(multi_val_data)

    #True
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    
    #False
    if multi_train_data is not None:
        multi_train_iterator = iter(multi_train_data)
    else:
        multi_train_iterator = None
    
    #True
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None
    
    #False
    if multi_val_data is not None:
        multi_val_iterator = iter(multi_val_data)
    else:
        multi_val_iterator = None
    
    
    iteration = 0
    #200000
    if args.train_iters > 0:
        #1
        if args.do_train:
            with ExitStack() as stack:
                
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)
                
                iteration, skipped = train(model, 
                                           optimizer,
                                           lr_scheduler,
                                           (train_data_iterator, multi_train_iterator),
                                           (val_data_iterator, multi_val_iterator),
                                           timers, args, 
                                           summary_writer=summary_writer
                                           )
        
        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, (val_data_iterator, multi_val_iterator),
                                                  model, args, timers, verbose=False, forward_step_func=forward_step)
    
    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    # if test_data is not None:
    #     test_data_iterator = iter(test_data)
    # else:
    #     test_data_iterator = None

    # if args.do_test:
    #     prefix = 'the end of training for test data'
    #     evaluate_and_print_results(prefix, (test_data_iterator, None),
    #                                model, args, timers, verbose=True, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
