import deepspeed
import oneflow as flow
from oneflow.optim import Adam
# from apex.optimizers import FusedAdam as Adam
# from oneflow.nn import Adam
# from torch import distributed as dist

import mpu
from fp16 import FP16_Module, FP16_Optimizer, DynamicLossScaler
from learning_rates import AnnealingLR
from model import GLMModel, glm_get_params_for_weight_decay_optimization
from model import GLMForMultiTokenCloze, GLMForMultiTokenClozeFast, GLMForSingleTokenCloze, GLMForSequenceClassification
from model import PyTorchDistributedDataParallel as TorchDDP, DistributedDataParallel as LocalDDP
from model.modeling_bert import BertForMultipleChoice, BertForSequenceClassification
from utils import print_rank_0, get_checkpoint_name, get_checkpoint_iteration


def load_pretrained(model, checkpoint_path, args, task_tokens=None):
    #checkpoint_path : /data/lichunyou/GLM/GLM_copa/copa_model/blank-base-copa_08-25-23-55
    #load_dir : /data/lichunyou/GLM/GLM_copa/copa_model/blank-base-copa_08-25-23-55
    #tag : best
    #release : False
    #sucess : True
    load_dir, tag, release, success = get_checkpoint_iteration(checkpoint_path)
    #/data/lichunyou/GLM/GLM_copa/copa_model/blank-base-copa_08-25-23-55/best/mp_rank_00_model_states.pt
    # checkpoint_name = get_checkpoint_name(load_dir, tag, release)
    checkpoint_name = '/dataset/lichunyou/GLM/GLM_copa_oneflow/other/copa_model/blank-base-copa_08-25-23-55/best/mp_rank_00_model_states.pt'
    
    # if mpu.get_data_parallel_rank() == 0:
    #     print('global rank {} is loading pretrained model {}'.format(
    #         torch.distributed.get_rank(), checkpoint_name))
    
    # #True
    # if args.deepspeed:
    #     model = model.module
    # #False
    # if isinstance(model, TorchDDP):
    #     model = model.module
    # #False
    # if isinstance(model, FP16_Module):
    #     model = model.module
    # #False
    # if hasattr(model, "model"):
    #     model = model.model

    
    def extend_embedding_weights(state_weights, model_weights):
        original_length = state_weights.shape[0]
        assert original_length <= args.max_position_embeddings + 1
        new_weights = model_weights.clone()
        new_weights[:original_length] = state_weights
        return new_weights
    
    import torch
    sd = torch.load(checkpoint_name, map_location='cpu')
    
    #True
    if args.block_lm:
        #True
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            position_weights = sd['module']["transformer.position_embeddings.weight"]
            #False
            if args.max_position_embeddings + 1 > position_weights.shape[0]:
                sd['module']["transformer.position_embeddings.weight"] = extend_embedding_weights(
                    position_weights, model.state_dict()["transformer.position_embeddings.weight"].data)
                print_rank_0(f"Extend position embedding to {args.max_position_embeddings + 1}")
        #True
        if "transformer.block_position_embeddings.weight" in sd["module"]:
            block_position_weights = sd['module']["transformer.block_position_embeddings.weight"]
            #False
            if args.max_position_embeddings + 1 > block_position_weights.shape[0]:
                sd['module']["transformer.block_position_embeddings.weight"] = extend_embedding_weights(
                    block_position_weights,
                    model.state_dict()["transformer.block_position_embeddings.weight"].data)
                print_rank_0(f"Extend block position embedding to {args.max_position_embeddings + 1}")
    
    torch_params = sd["module"]
    flow_params = {}
    for k in torch_params.keys():
        flow_params[k] = flow.Tensor(torch_params[k].numpy().astype("float32"))
    model.load_state_dict(flow_params)

    # missing_keys, unexpected_keys = model.load_state_dict(sd['module'], strict=False)

    # if missing_keys or unexpected_keys:
    #     print_rank_0(f"Missing keys {missing_keys}, unexpected keys {unexpected_keys}")
    # #False
    # if args.continuous_prompt and args.prompt_init:
    #     model.prompt_spell.init_embedding(model.word_embeddings.weight.data, task_tokens)


def get_model(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):
    #args.pretrained_bert : False,  model_type: multiple_choice
    #args.pretrained_bert : False,  model_type: None
    if args.pretrained_bert:
        if model_type == "multiple_choice":
            model = BertForMultipleChoice.from_pretrained(args.tokenizer_model_type,
                                                          cache_dir=args.cache_dir,
                                                          fp32_layernorm=args.fp32_layernorm,
                                                          fp32_embedding=args.fp32_embedding,
                                                          layernorm_epsilon=args.layernorm_epsilon)
        elif model_type == "classification":
            model = BertForSequenceClassification.from_pretrained(args.tokenizer_model_type,
                                                                  cache_dir=args.cache_dir,
                                                                  fp32_layernorm=args.fp32_layernorm,
                                                                  fp32_embedding=args.fp32_embedding,
                                                                  layernorm_epsilon=args.layernorm_epsilon,
                                                                  num_labels=num_labels)
        else:
            raise NotImplementedError
    else:
        output_predict, paralle_output = True, True
        if (model_type == "multiple_choice" or model_type == "classification") and not args.cloze_eval:
            output_predict = False   # True,true
        if model_type is not None:
            paralle_output = False   # False,true
        if spell_length is not None:
            print_rank_0(f"Continuous spell length {spell_length}")
        
        # print(args.num_layers)
        # print(args.vocab_size)
        # print(args.hidden_size)
        # print(args.num_attention_heads)
        # print(args.hidden_dropout)
        # print(args.attention_dropout)
        # print(args.hidden_dropout)
        # print(args.max_position_embeddings)
        # print(args.mem_length)
        # print(args.checkpoint_activations)
        # print(args.checkpoint_num_layers)
        # print(paralle_output)
        # print(args.transformer_xl)
        # print(args.block_lm and not args.masked_lm)
        # print(output_predict)
        # print(spell_length)
        # print(args.prompt_func)
        # print(args.attention_scale)
        # exit(0)

        model = GLMModel(num_layers=args.num_layers,
                         vocab_size=args.vocab_size,
                         hidden_size=args.hidden_size,
                         num_attention_heads=args.num_attention_heads,
                         embedding_dropout_prob=args.hidden_dropout,
                         attention_dropout_prob=args.attention_dropout,
                         output_dropout_prob=args.hidden_dropout,
                         max_sequence_length=args.max_position_embeddings,
                         max_memory_length=args.mem_length,
                         checkpoint_activations=args.checkpoint_activations,
                         checkpoint_num_layers=args.checkpoint_num_layers,
                         parallel_output=paralle_output,
                         relative_encoding=args.transformer_xl,
                         block_position_encoding=args.block_lm and not args.masked_lm,
                         output_predict=output_predict,
                         spell_length=spell_length,
                         spell_func=args.prompt_func,
                         attention_scale=args.attention_scale)
        #False
        if args.freeze_transformer:
            model.freeze_transformer(tune_prefix_layers=args.tune_prefix_layers)
        #model_type:multiple_choice
        #model_type:none
        if model_type is not None:
            #True
            if model_type == 'multiple_choice':
                #True
                if args.cloze_eval:
                    #True
                    if multi_token:
                        #False
                        if args.fast_decode:
                            model = GLMForMultiTokenClozeFast(model, length_penalty=args.length_penalty)
                        else:

                            model = GLMForMultiTokenCloze(model, length_penalty=args.length_penalty)
                    else:
                        model = GLMForSingleTokenCloze(model, take_softmax=args.adapet)
                else:
                    model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                         num_class=num_labels)
            elif model_type == 'classification':
                model = GLMForSequenceClassification(model, args.hidden_size, args.output_dropout, args.pool_token,
                                                     num_class=num_labels)
            elif model_type == 'generation':
                pass
            else:
                raise NotImplementedError(model_type)
    #True
    # if mpu.get_data_parallel_rank() == 0:
    #     print(' > number of parameters on model parallel rank {}: {}'.format(
    #         mpu.get_model_parallel_rank(),
    #         sum([p.nelement() for p in model.parameters()])), flush=True)

    #False
    # if args.fp16:
    #     model.half()

    model.cuda()
    
    #true
    # if args.fp16:
    #     model = FP16_Module(model)
    
    #False
    # if not args.deepspeed and (args.train_iters or args.epochs):
    #     if args.DDP_impl == 'torch':
    #         i = flow.cuda.current_device()
    #         model = TorchDDP(model, device_ids=[i], output_device=i,
    #                          process_group=mpu.get_data_parallel_group())
    #     elif args.DDP_impl == 'local':
    #         model = LocalDDP(model)
    #     else:
    #         print_rank_0("Skip DDP model")
    return model


def get_optimizer_param_groups(model):
    #False
    while isinstance(model, (LocalDDP, TorchDDP, FP16_Module)):
        model = model.module
    param_groups = glm_get_params_for_weight_decay_optimization(model)
    
    # for param_group in param_groups:
    #     for param in param_group['params']:
    #         if not hasattr(param, 'model_parallel'):
    #             param.model_parallel = False

    return param_groups


def get_optimizer(param_groups, args):
    #False
    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = flow.optim.AdamW
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        #True
        if args.optimizer == 'adam':
            optimizer = Adam(param_groups,
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             betas=(args.adam_beta1, args.adam_beta2),
                             eps=args.adam_eps)
        elif args.optimizer == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(param_groups, lr=args.lr, relative_step=False, warmup_init=False)
        else:
            raise NotImplementedError

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if hasattr(args, "deepspeed") and args.deepspeed:
        raise NotImplementedError
      
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):

    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    if args.finetune:
        num_iters = num_iters // args.gradient_accumulation_steps
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters - warmup_iter,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio)

    return lr_scheduler


def setup_model_and_optimizer(args, model_type=None, multi_token=True, num_labels=None, spell_length=None):

    model = get_model(args, model_type=model_type, multi_token=multi_token, num_labels=num_labels,
                      spell_length=spell_length)
    param_groups = get_optimizer_param_groups(model)
    
    #False
    #True
    if args.train_data is not None or args.data_dir is not None and (args.epochs > 0 or args.train_iters > 0):
        #False
        if args.deepspeed:
            print_rank_0("DeepSpeed is enabled.")

            # model, optimizer, _, _ = deepspeed.initialize(
            #     model=model,
            #     model_parameters=param_groups,
            #     args=args,
            #     mpu=mpu,
            #     dist_init_required=False
            # )
            optimizer = Adam(param_groups,
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             betas=(args.adam_beta1, args.adam_beta2),
                             eps=args.adam_eps)
        else:
            # optimizer = get_optimizer(param_groups, args)
            optimizer =  Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay,
                             betas=(args.adam_beta1, args.adam_beta2),
                             eps=args.adam_eps)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
    else:
        optimizer, lr_scheduler = None, None
   
    return model, optimizer, lr_scheduler


def backward_step(optimizer, model, lm_loss, args, timers):
   
    loss = lm_loss
    
    #False
    if args.deepspeed:
        # optimizer.zero_grad()
        # model.backward(loss)
        loss.backward()
    else:
        optimizer.zero_grad()
        #False
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
          
            loss.backward()
    
    if True:
    # if args.deepspeed or args.DDP_impl == 'torch':
        timers('allreduce').reset()
    else:
        timers('allreduce').start()
        model.allreduce_params(reduce_after=False, fp32_allreduce=args.fp32_allreduce)
        timers('allreduce').stop()
    
    #True
    if not args.deepspeed:
        #False
        if args.fp16:
            optimizer.update_master_grads()
    
        #true
        if args.clip_grad > 0:
            #True
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", flow.cuda.memory_allocated() / (1024 * 1024 * 1024), "GigaBytes")
        print("Max Memory Allocated ", flow.cuda.max_memory_allocated() / (1024 * 1024 * 1024), "GigaBytes")
        print("Cache Allocated ", flow.cuda.memory_cached() / (1024 * 1024 * 1024), "GigaBytes")
        print("Max cache Allocated ", flow.cuda.max_memory_cached() / (1024 * 1024 * 1024), "GigaBytes")
        print(" ")


def train_step(data_iterator, model, optimizer, lr_scheduler, args, timers, forward_step_func, mems=None,
               single_step=False):
    lm_loss_total, count = 0.0, 0
    mems = [] if mems is None else mems

    #False
    if not args.deepspeed:
        optimizer.zero_grad()
    while True:
        skipped_iter, complete = 0, False
        
        timers('forward').start()
        lm_loss, mems, _ = forward_step_func(data_iterator, model, args, timers, mems)
        timers('forward').stop()
        
        #False
        if not args.deepspeed:
            lm_loss /= args.gradient_accumulation_steps

        reduced_loss = lm_loss.detach().clone().view(1)
        # flow.distributed.all_reduce(reduced_loss.data, group=mpu.get_data_parallel_group())
        reduced_loss.data = reduced_loss.data / (args.world_size / args.model_parallel_size)
    
        #True
        if not DynamicLossScaler._has_inf_or_nan(reduced_loss):
            lm_loss_total += reduced_loss
            count += 1

            timers('backward').start()
            backward_step(optimizer, model, lm_loss, args, timers)
            timers('backward').stop()
            
            timers('optimizer').start()
            
            # optimizer.zero_grad()
            optimizer.step()
            complete = True
            lr_scheduler.step()
            
            # if args.deepspeed:
            #     if model.is_gradient_accumulation_boundary():
            #         model.step()
            #         complete = True
            #         if not (args.fp16 and optimizer.overflow):
            #             lr_scheduler.step()
            #         else:
            #             skipped_iter = 1
            #     else:
            #         model.step()
            # else:
            #     if count == args.gradient_accumulation_steps:
            #         optimizer.step()
            #         complete = True
                    
            #         if not (args.fp16 and optimizer.overflow):
            #             lr_scheduler.step()
            #         else:
            #             skipped_iter = 1
       
            timers('optimizer').stop()
            if complete:
                break
        else:
            print_rank_0("Found NaN loss, skip backward")
            del lm_loss, reduced_loss
            mems = []
        
        if single_step:
            break
    if args.deepspeed:
        lm_loss_total = lm_loss_total / count
    return lm_loss_total, skipped_iter, mems
