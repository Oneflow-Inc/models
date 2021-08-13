import os
import shutil
import json

import numpy as np
import torch
import oneflow as flow
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from models.roberta import Roberta
from roberta_weight_utils import Roberta_trans, colored_string

TO_CUDA = False


def get_model(pretrained='roberta-base'):

    colored_string("Generating roberta with transformers, pretrained = {}...".format(
        pretrained), color="green", end="")

    tokenizer = RobertaTokenizer.from_pretrained(pretrained)
    roberta_trans = RobertaModel.from_pretrained(pretrained)

    colored_string("Done.", color="green")
    colored_string("Generating roberta with oneflow...", color="green", end="")

    config = roberta_trans.config
    kwargs = {
        "vocab_size": config.vocab_size,
        "type_vocab_size": config.type_vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "chunk_size_feed_forward": 0,
        "num_layers": config.num_hidden_layers,
        "nheads": config.num_attention_heads,
        "activation": config.hidden_act,
        "pad_token_id": config.pad_token_id,
        "layer_norm_eps": config.layer_norm_eps,
        "attn_dropout": config.attention_probs_dropout_prob,
        "hidden_dropout": config.hidden_dropout_prob,
        "position_embedding_type": config.position_embedding_type,
        "is_decoder": config.is_decoder,
        "add_pooling_layer": True,
        "add_cross_attention": config.add_cross_attention
    }
    roberta_flow = Roberta(**kwargs)
    colored_string("Done.", color="green")

    return roberta_flow, roberta_trans, kwargs


# Calculate L1Loss in numpy
def L1Loss_numpy(flow_tensor, torch_tensor):

    if torch_tensor.device == torch.cuda:
        torch_tensor = torch_tensor.cpu()

    return np.mean(flow_tensor.numpy() - torch_tensor.detach().numpy())


def test_model(roberta_trans, pretrain_dir, pretrained_model):

    save_dir = os.path.join(pretrain_dir, pretrained_model)
    with open(os.path.join(save_dir, pretrained_model+".json"), mode="r") as f:
        kwargs = json.load(f)
    roberta_flow = Roberta(**kwargs)
    roberta_flow.load_state_dict(flow.load(save_dir))

    config = roberta_trans.config

    colored_string("Testing outputs...", color="green")

    if TO_CUDA:
        roberta_flow = roberta_flow.to("cuda")
        roberta_trans = roberta_trans.to("cuda")

    roberta_trans.eval()
    roberta_flow.eval()

    bs = 32
    seq_len = 128

    def get_random_tensor(low, high, sz, cuda=TO_CUDA):
        arr = np.random.randint(low=low, size=sz, high=high)
        if cuda:
            flow_tensor = flow.Tensor(arr).to("cuda")
            torch_tensor = torch.Tensor(arr).cuda()
        else:
            flow_tensor = flow.Tensor(arr)
            torch_tensor = torch.Tensor(arr)
        return flow_tensor, torch_tensor

    # Set inputs
    input_ids = get_random_tensor(0, 2000, (bs, seq_len))
    attention_mask = get_random_tensor(0, 1, sz=(bs, seq_len))
    token_type_ids = get_random_tensor(0, 1, (bs, seq_len))
    position_ids = get_random_tensor(
        1, config.max_position_embeddings - 1, (bs, seq_len))
    head_mask = get_random_tensor(
        0, 1, (config.num_hidden_layers, config.num_attention_heads))
    inputs_embeds = None
    output_attentions = True
    output_hidden_states = True
    encoder_hidden_states = get_random_tensor(
        0, 5, (bs, seq_len, config.hidden_size))
    encoder_attention_mask = get_random_tensor(0, 5, (bs, seq_len))
    past_key_values = tuple([
        tuple([
            get_random_tensor(
                0, 1, (bs, config.num_attention_heads, seq_len, seq_len))
            for i in range(4)
        ])
        for j in range(config.num_hidden_layers)
    ])
    use_cache = False
    past_key_values = None

    # Run forward
    # return:
    # sequence_output, (pooled_output, past_key_values, hidden_states, attentions, cross_attentions)
    colored_string("Running roberta with oneflow...", color="green", end="")
    seq_flow, pool_flow = roberta_flow(input_ids[0].to(flow.int32), attention_mask[0], token_type_ids[0].to(flow.int32), position_ids[0].to(flow.int32), head_mask[0],
                                       inputs_embeds, encoder_hidden_states[0], encoder_attention_mask[0], past_key_values, use_cache, output_attentions, output_hidden_states)
    pool_flow, pkv_flow, hidden_flow, attn_flow, cross_attn_flow = pool_flow
    colored_string("Done.", color="green")
    colored_string("Running roberta with transformers...",
                   color="green", end="")
    output = roberta_trans(input_ids[1].to(torch.long), attention_mask[1], token_type_ids[1].to(torch.long), position_ids[1].to(
        torch.long), head_mask[1], inputs_embeds, encoder_hidden_states[1], encoder_attention_mask[1], past_key_values, use_cache, output_attentions, output_hidden_states, return_dict=True)
    seq_trans = output.last_hidden_state
    pool_trans = output.pooler_output
    pkv_trans = output.past_key_values
    hidden_trans = output.hidden_states
    attn_trans = output.attentions
    cross_attn_trans = output.cross_attentions
    colored_string("Done.", color="green")

    # Calculate errors
    colored_string("Calculating errors...", color="green")
    seq_error = L1Loss_numpy(seq_flow, seq_trans)
    pool_error = L1Loss_numpy(pool_flow, pool_trans)
    colored_string("Sequence output error:{}".format(
        seq_error.item()), color="green")
    colored_string("Pooled output error:{}".format(
        pool_error.item()), color="green")
    colored_string("Done.", color="green")

    return seq_error, pool_error


def save_model(model_flow, kwargs, pretrain_dir, pretrained_model):

    save_dir = os.path.join(pretrain_dir, pretrained_model)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        flow.save(model_flow.state_dict(), save_dir)
        fp = open(os.path.join(save_dir, pretrained_model + ".json"), mode="w")
        json.dump(kwargs, fp)
        fp.close()
    else:
        colored_string(
            "Model save directory '{}' already exists. Do you still want to save? (y/n)".format(save_dir), color="blue")
        ans = input()
        while ans.lower() != "y" and ans.lower() != "n":
            ans = input("Please input y/n:")
        if ans.lower() == "y":
            shutil.rmtree(save_dir)
            assert not os.path.exists(save_dir)
            os.mkdir(save_dir)
            flow.save(model_flow.state_dict(), save_dir)
            fp = open(os.path.join(save_dir, pretrained_model + ".json"), mode="w")
            json.dump(kwargs, fp)
            fp.close()


if __name__ == '__main__':

    pretrain_dir = "roberta_pretrain_oneflow"
    pretrained_model = "roberta-large"

    roberta_flow, roberta_trans, kwargs = get_model(pretrained_model)

    colored_string("Transforming weights...", color="green")
    roberta_flow = Roberta_trans(roberta_flow, roberta_trans)
    colored_string("Done.", color="green")

    save_model(roberta_flow, kwargs, pretrain_dir, pretrained_model)

    test_model(roberta_trans, pretrain_dir, pretrained_model)
