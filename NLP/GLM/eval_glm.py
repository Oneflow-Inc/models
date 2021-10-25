"""Finetune utilities."""

import os
import json

import random

from tasks.data_utils import build_data_loader, FakeDataloader
from utils import get_sample_writer, get_log_dir, print_and_save_args, debug_finetune_data
from arguments import get_args
from filelock import FileLock
import pretrain_glm
from pretrain_glm import forward_step as lm_forward_step
import pathlib
import mpu


import oneflow as flow
import oneflow.utils.data
from configure_data import prepare_tokenizer

from utils import print_rank_0
from utils import Timers
from train_utils import setup_model_and_optimizer, train_step, load_pretrained
from utils import load_checkpoint, save_checkpoint
from pretrain_glm import report_iteration_metrics
from pretrain_glm import evaluate_and_print_results
from pretrain_glm import initialize_distributed
from pretrain_glm import set_random_seed
from configure_data import make_data_loader



tokenizer = None


def finetune(args, 
             train_valid_datasets_provider, #tasks/superglue/finetune.py:train_valid_datasets_provider
             model_kwargs, 
             end_of_epoch_callback_provider=None #metrics_func_provider
             ):
    
    global tokenizer
    timers = Timers()

    tokenizer = prepare_tokenizer(args)
    pretrain_glm.tokenizer = tokenizer

    if args.save:
        args.save = os.path.join(args.save, args.experiment_name) 
    #args.save : /data/lichunyou/GLM/GLM_copa/copa_model/blank-base-copa_09-13-16-15
   
    timers('train/valid/test dataset/dataloder').start()
    train_dataloader, valid_dataloader = None, None
    train_block_dataloader, valid_block_dataloader = None, None

    timers('train/valid/test dataset/dataloder').stop()
    
    timers('callback function').start()
    end_of_epoch_callback, end_of_train_callback = None, None

    #True
    if end_of_epoch_callback_provider is not None:
        #False
        if train_valid_datasets_provider is not None and args.epochs > 0 and not args.no_validation:
            end_of_epoch_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=False)
        #tasks/eval_utils.py:metrics_func
        end_of_train_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=True)
    timers('callback function').stop()

    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, **model_kwargs)

    timers('model and optimizer').stop()

    timers('pretrained checkpoint').start()
    #True
    if args.load_pretrained is not None and not args.pretrained_bert:
        task_tokens = None
        with FileLock(os.path.join(pathlib.Path.home(), "checkpoint_lock"), timeout=-1):
            load_pretrained(model, args.load_pretrained, args, task_tokens=task_tokens)

    timers('pretrained checkpoint').stop()
    args.iteration = 0
    summary_writer = None
    #True
    if True:
        args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
        #False
        if os.path.exists(os.path.join(args.log_dir, "test_results.json")) and args.load is None and not args.overwrite:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.log_dir))
        summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)
    
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    score_dict = None
    
    #True: tasks/eval_utils.py:metrics_func
    if end_of_train_callback is not None:
        score_dict = end_of_train_callback(model, epoch=-1, output_predictions=True)
    # if score_dict is not None and torch.distributed.get_rank() == 0:
    if score_dict is not None:
        score_dict.update({"type": "test"})
        with open(os.path.join(args.log_dir, "test_results.json"), "w") as output:
            output.write(json.dumps(score_dict) + "\n")

    print_rank_0('done :-)')


if __name__ == '__main__':
    
    flow.backends.cudnn.enabled = False

    args = get_args()
    assert args.finetune

    # initialize_distributed(args)
    # set_random_seed(args.seed)

    from tasks.superglue.dataset import PROCESSORS

    superglue_tasks = list(PROCESSORS.keys())

    #True
    if args.task.lower() in superglue_tasks:
        from tasks.superglue.finetune import main
    elif args.task.lower() in ['lambda', 'wikitext', 'language_model']:
        from tasks.language_model.finetune import main
    elif args.task.lower() in ['cnn_dm', 'cnn_dm_original', 'gigaword', 'blank', 'squad_generation', 'xsum', 'extraction']:
        from tasks.seq2seq.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    main(args)
