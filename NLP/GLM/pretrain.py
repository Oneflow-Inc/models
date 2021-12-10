import time
import oneflow as flow
import oneflow.nn as nn
from oneflow.nn.parallel import DistributedDataParallel as ddp
from data import prepare_tokenizer, make_dataset
from args import get_args
from modeling import GLMModel
import sys
sys.path.append(".")


def load_torch_model(model, path):
    import torch
    torch_params = torch.load(path, map_location='cpu')
    flow_params = {}
    for k in torch_params.keys():
        flow_params[k] = flow.Tensor(
            torch_params[k].numpy().astype("float32"))
    model.load_state_dict(flow_params, strict=False)
    print("load pretraining model succeed!")


class GLMGraph(nn.Graph):
    def __init__(self, args, model, optimizer, lr_scheduler):
        super().__init__()
        self.glm = model
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        if args.graph_fp16:
            print("using amp fp16!!")
            self.config.enable_amp(True)
            grad_scaler = flow.amp.GradScaler(
                init_scale=2 ** 30,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
            )
            self.set_grad_scaler(grad_scaler)

    def build(self, tokens, position_ids, attention_mask, labels, loss_mask):

        logits, *mems = self.glm(tokens, position_ids, attention_mask)
        losses = flow._C.sparse_softmax_cross_entropy(logits, labels)

        loss_mask = loss_mask.view((-1,))
        loss = flow.sum(losses.view((-1,)) * loss_mask)

        loss = loss / loss_mask.sum()
        loss.backward()
        return loss


def get_model(args):
    assert args.mode in ["eager", "graph"]

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
                     parallel_output=True,
                     relative_encoding=args.transformer_xl,
                     block_position_encoding=args.block_lm and not args.masked_lm,
                     output_predict=True,
                     spell_length=None,
                     spell_func=args.prompt_func,
                     attention_scale=args.attention_scale)

    if args.debug_loss:
        # load pretrain
        load_torch_model(model, path=args.debug_pretrain_model)
        model.eval()
    else:
        model.train()

    if args.mode == "graph":
        placement = flow.env.all_device_placement("cuda")
        model = model.to_consistent(placement=placement,
                                    sbp=flow.sbp.broadcast
                                    )
    elif args.mode == "eager":
        model.cuda()
        if flow.env.get_world_size() > 1:
            model = ddp(model)
    else:
        raise NotImplementedError

    optimizer = flow.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                betas=(args.adam_beta1, args.adam_beta2),
                                eps=args.adam_eps)
    lr_scheduler = flow.optim.lr_scheduler.StepLR(optimizer, step_size=100000)

    if args.mode == "eager":
        return model, optimizer, lr_scheduler
    if args.mode == "graph":
        graph_model = GLMGraph(args, model, optimizer, lr_scheduler)
        return graph_model, None, None
    else:
        raise NotImplementedError


def get_dataloader(args):
    tokenizer = prepare_tokenizer(args)
    args.data_set_type = "Block"
    train_data, val_data, test_data = make_dataset(args, tokenizer)
    train_data_iterator = iter(train_data)

    return train_data_iterator


def train_step(data_iterator, model, optimizer, lr_scheduler):
    data = next(data_iterator)
    tokens = data['text'].long().cuda()
    labels = data['target'].long().cuda()
    attention_mask = data['attention_mask'].long().cuda()
    loss_mask = data['loss_mask'].float().cuda()
    position_ids = data['position_id'].long().cuda()

    if isinstance(model, nn.Graph):
        placement = flow.env.all_device_placement("cuda")
        sbp = flow.sbp.split(0)

        tokens = tokens.to_consistent(placement=placement, sbp=sbp)
        position_ids = position_ids.to_consistent(placement=placement, sbp=sbp)
        attention_mask = attention_mask.to_consistent(
            placement=placement, sbp=sbp)
        labels = labels.to_consistent(placement=placement, sbp=sbp)
        loss_mask = loss_mask.to_consistent(placement=placement, sbp=sbp)

        loss = model(tokens, position_ids, attention_mask, labels, loss_mask)
    else:
        optimizer.zero_grad()

        logits = model(tokens, position_ids, attention_mask)[0]
        losses = flow._C.sparse_softmax_cross_entropy(logits, labels)
        loss_mask = loss_mask.view((-1,))
        loss = flow.sum(losses.view((-1,)) * loss_mask)
        if loss_mask.sum().item() > 0:
            loss = loss / loss_mask.sum()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return loss


if __name__ == "__main__":
    args = get_args()
    if args.debug_loss:
        args.hidden_dropout = 0.0
        loss_txt = open(args.loss_txt_path, "w")
        
    if args.mode == "graph":
            flow.boxing.nccl.set_fusion_threshold_mbytes(16)
            flow.boxing.nccl.set_fusion_max_ops_num(24)
            # flow.boxing.nccl.enable_use_compute_stream(True)

    model, optimizer, lr_scheduler = get_model(args)
    train_data_iterator = get_dataloader(args)
    
    # warm up 10 iters
    if not args.debug_loss:
        for _ in range(10):
            loss = train_step(train_data_iterator, model, optimizer, lr_scheduler)
    loss.numpy() # sync cuda
    
    count = 0
    tb = time.time()
    t0 = time.time()

    for _ in range(args.train_iters):
        loss = train_step(train_data_iterator, model, optimizer, lr_scheduler)
        
        count += 1
        if count % args.print_iter == 0:
            loss.numpy() # sync cuda
            t1 = time.time()
            total_batch_size = flow.env.get_world_size() * \
                               args.batch_size * \
                               args.print_iter
            though_out = total_batch_size / (t1 - t0)
            t0 = time.time()
            if flow.env.get_rank() == 0:
                print(f"iter: {count}, though_out: {though_out}")
                
        if args.debug_loss:
            loss_txt.write(str(loss.item())+"\n")

    te = time.time()
    total_batch_size = flow.env.get_world_size() * \
                       args.batch_size * \
                       args.train_iters
    avg_though_out = total_batch_size / (te - tb)
    if flow.env.get_rank() == 0:
        print(f"avg_though_out: {avg_though_out}, total time: {te - tb}s")
