import json
import os
import shutil
import sys

import oneflow as flow
import numpy as np
import torch
import transformers

from base_weight_utils import colored_string
from roberta_weight_utils import (
    Roberta_trans,
    RobertaForMaskedLM_trans,
    RobertaForSequenceClassification_trans
)
sys.path.append("../")
from models.roberta import (
    Roberta,
    RobertaForMaskedLM,
    RobertaForSequenceClassification
)


# model saves in `./save_dir/model_dir/weights`
# parameters saved in `./save_dir/model_dir/parameters.json`
class BaseTransform:
    def __init__(self, model_flow, model_torch, pretrained_model, save_dir, model_dir, trans_func, cuda=False):

        self.save_dir = os.path.join(save_dir, model_dir)
        self.weights_dir = os.path.join(self.save_dir, "weights")
        self.param_path = os.path.join(self.save_dir, "parameters.json")
        self.pretrained_model = pretrained_model

        self.config = transformers.RobertaConfig.from_pretrained(pretrained_model)
        self.device = "cuda" if cuda else "cpu"
        self.build_params()
        self.build_model(model_flow, model_torch)
        self.trans_func = trans_func

    def build_params(self):

        self.kwargs = {
            "vocab_size": self.config.vocab_size,
            "type_vocab_size": self.config.type_vocab_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "hidden_size": self.config.hidden_size,
            "intermediate_size": self.config.intermediate_size,
            "chunk_size_feed_forward": 0,
            "num_layers": self.config.num_hidden_layers,
            "nheads": self.config.num_attention_heads,
            "activation": self.config.hidden_act,
            "pad_token_id": self.config.pad_token_id,
            "layer_norm_eps": self.config.layer_norm_eps,
            "attn_dropout": self.config.attention_probs_dropout_prob,
            "hidden_dropout": self.config.hidden_dropout_prob,
            "position_embedding_type": self.config.position_embedding_type,
            "is_decoder": self.config.is_decoder,
            "add_cross_attention": self.config.add_cross_attention,
        }

    def build_model(self, model_flow, model_torch):

        colored_string("Generating model with transformers, pretrained = {}...".format(self.pretrained_model), color="green", end="")

        self.model_torch = model_torch.from_pretrained(self.pretrained_model).to(self.device)

        colored_string("Done.", color="green")
        colored_string("Generating model with oneflow...", color="green", end="")

        self.model_flow = model_flow(**self.kwargs).to(self.device)

        colored_string("Done.", color="green")

    def run(self, test_args, test_only=False):

        if not test_only:
            self.transform()
            self.save()
        self.test(**test_args)

    def transform(self):

        colored_string("Transforming weights...", color="green")
        self.model_flow = self.trans_func(self.model_flow, self.model_torch)
        colored_string("Done.", color="green")   

    def test(self, bs, seq_len):

        raise NotImplementedError 
    
    def L1Loss_numpy(self, flow_tensor, torch_tensor):

        if torch_tensor.device == torch.cuda:
            torch_tensor = torch_tensor.cpu()

        return np.mean(flow_tensor.numpy() - torch_tensor.detach().numpy())

    def get_random_tensor(self, low, high, sz, if_int=False):

        arr = np.random.randint(low=low, size=sz, high=high)
        flow_tensor = flow.tensor(arr, device=self.device)
        torch_tensor = torch.tensor(arr, device=self.device)

        return (flow_tensor.to(flow.int64), torch_tensor.to(torch.long)) if if_int \
                else (flow_tensor, torch_tensor)

    def save_param(self):

        with open(self.param_path, mode="w") as fp:
                json.dump(self.kwargs, fp)
        
    def save(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.weights_dir)
            flow.save(self.model_flow.state_dict(), self.weights_dir)
            self.save_param()
            colored_string("Model saved.", color="green")
        else:
            colored_string(
                "Model save directory '{}' already exists. Do you still want to save? (y/n)".format(self.save_dir), color="blue")
            ans = input()
            while ans.lower() != "y" and ans.lower() != "n":
                ans = input("Please input y/n:")
            if ans.lower() == "y":
                shutil.rmtree(self.save_dir)
                assert not os.path.exists(self.save_dir)
                os.makedirs(self.weights_dir)
                flow.save(self.model_flow.state_dict(), self.weights_dir)
                self.save_param()
                colored_string("Model saved.", color="green")


class RobertaTransform(BaseTransform):
    def __init__(self, pretrained_model, save_dir, model_dir, cuda=False):
        colored_string("Transform weights of Roberta", color="green")
        super().__init__(Roberta, transformers.RobertaModel, pretrained_model,
                        save_dir, model_dir, Roberta_trans, cuda)

    def test(self, bs, seq_len):

        with open(self.param_path, mode="r") as f:
            kwargs = json.load(f)
        self.model_flow = Roberta(**kwargs)
        self.model_flow.load_state_dict(flow.load(self.weights_dir))

        colored_string("Testing outputs...", color="green")

        self.model_torch.eval()
        self.model_flow.eval()

        # Set inputs
        input_ids = self.get_random_tensor(0, 2000, (bs, seq_len), if_int=True)
        attention_mask = self.get_random_tensor(0, 2, sz=(bs, seq_len))
        token_type_ids = self.get_random_tensor(0, self.config.type_vocab_size, (bs, seq_len), if_int=True)
        position_ids = self.get_random_tensor(
            1, self.config.max_position_embeddings - 1, (bs, seq_len), if_int=True)
        head_mask = self.get_random_tensor(
            0, 2, (self.config.num_hidden_layers, self.config.num_attention_heads))
        inputs_embeds = None
        output_attentions = True
        output_hidden_states = True
        encoder_hidden_states = self.get_random_tensor(
            0, 5, (bs, seq_len, self.config.hidden_size))
        encoder_attention_mask =self.get_random_tensor(0, 2, (bs, seq_len))
        use_cache = False
        past_key_values = None

        # Run forward
        colored_string("Running model with oneflow...", color="green", end="")
        out_flow = self.model_flow(input_ids[0], attention_mask[0], token_type_ids[0], position_ids[0], head_mask[0], 
                            inputs_embeds, encoder_hidden_states[0], encoder_attention_mask[0], past_key_values, 
                            use_cache, output_attentions, output_hidden_states)
        seq_flow, pool_flow, pkv_flow, hidden_flow, attn_flow, cross_attn_flow = out_flow
        colored_string("Done.", color="green")

        colored_string("Running model with transformers...", color="green", end="")
        output = self.model_torch(input_ids[1], attention_mask[1], token_type_ids[1], position_ids[1], head_mask[1], 
                                inputs_embeds, encoder_hidden_states[1], encoder_attention_mask[1], past_key_values, 
                                use_cache, output_attentions, output_hidden_states, return_dict=True)
        seq_trans = output.last_hidden_state
        pool_trans = output.pooler_output
        pkv_trans = output.past_key_values
        hidden_trans = output.hidden_states
        attn_trans = output.attentions
        cross_attn_trans = output.cross_attentions
        colored_string("Done.", color="green")

        # Calculate errors
        colored_string("Calculating errors...", color="green")
        seq_error = self.L1Loss_numpy(seq_flow, seq_trans)
        pool_error = self.L1Loss_numpy(pool_flow, pool_trans)
        colored_string("Sequence output error:{}".format(
            seq_error.item()), color="green")
        colored_string("Pooled output error:{}".format(
            pool_error.item()), color="green")
        colored_string("Done.", color="green")


class RobertaForMaskedLMTransform(BaseTransform):
    def __init__(self, pretrained_model, save_dir, model_dir, cuda=False):
        colored_string("Transform weights of RobertaForMaskedLM", color="green")
        super().__init__(RobertaForMaskedLM, transformers.RobertaForMaskedLM, pretrained_model, 
                        save_dir, model_dir, RobertaForMaskedLM_trans, cuda)

    def test(self, bs, seq_len):
        with open(self.param_path, mode="r") as f:
            kwargs = json.load(f)
        self.model_flow = RobertaForMaskedLM(**kwargs)
        self.model_flow.load_state_dict(flow.load(self.weights_dir))

        colored_string("Testing outputs...", color="green")

        self.model_flow.eval()
        self.model_torch.eval()

        # Set inputs
        input_ids = self.get_random_tensor(0, 2000, (bs, seq_len), if_int=True)
        attention_mask = self.get_random_tensor(0, 2, sz=(bs, seq_len))
        token_type_ids = self.get_random_tensor(0, self.config.type_vocab_size, (bs, seq_len), if_int=True)
        position_ids = self.get_random_tensor(
            1, self.config.max_position_embeddings - 1, (bs, seq_len), if_int=True)
        head_mask = self.get_random_tensor(
            0, 2, (self.config.num_hidden_layers, self.config.num_attention_heads))
        labels = self.get_random_tensor(0, self.config.vocab_size, (bs, seq_len))
        inputs_embeds = None
        output_attentions = True
        output_hidden_states = True
        encoder_hidden_states = self.get_random_tensor(
            0, 5, (bs, seq_len, self.config.hidden_size))
        encoder_attention_mask =self.get_random_tensor(0, 2, (bs, seq_len))

        # Run forward
        colored_string("Running model with oneflow...", color="green", end="")
        out_flow = self.model_flow(input_ids[0], attention_mask[0], token_type_ids[0], position_ids[0], head_mask[0], 
                            inputs_embeds, encoder_hidden_states[0], encoder_attention_mask[0], labels[0], output_attentions, output_hidden_states)
        loss_flow, scores_flow, pkv_flow, hidden_flow, attn_flow, cross_attn_flow = out_flow
        colored_string("Done.", color="green")

        colored_string("Running model with transformers...", color="green", end="")
        output = self.model_torch(input_ids[1], attention_mask[1], token_type_ids[1], position_ids[1], head_mask[1], 
                            inputs_embeds, encoder_hidden_states[1], encoder_attention_mask[1], labels[1], output_attentions, output_hidden_states, return_dict=True)
        loss_torch = output.loss
        scores_torch = output.logits
        colored_string("Done.", color="green")

        # Calculate errors
        colored_string("Calculating errors...", color="green")
        loss_error = self.L1Loss_numpy(loss_flow, loss_torch)
        scores_error = self.L1Loss_numpy(scores_flow, scores_torch)
        colored_string("Loss error:{}".format(
            loss_error.item()), color="green")
        colored_string("Logits error:{}".format(
            scores_error.item()), color="green")
        colored_string("Done.", color="green")


class RobertaForSequenceClassificationTransform(BaseTransform):
    def __init__(self, pretrained_model, save_dir, model_dir, cuda=False):
        colored_string("Transform weights of RobertaForSequenceClassification", color="green")
        super().__init__(RobertaForSequenceClassification, transformers.RobertaForSequenceClassification, pretrained_model,
                        save_dir, model_dir, RobertaForSequenceClassification_trans, cuda)

    def build_params(self):
        super().build_params()
        self.kwargs["num_labels"]  = self.config.num_labels
        self.kwargs["problem_type"] = self.config.problem_type

    def test(self, bs, seq_len):
        with open(self.param_path, mode="r") as f:
            kwargs = json.load(f)
        self.model_flow = RobertaForSequenceClassification(**kwargs)
        self.model_flow.load_state_dict(flow.load(self.weights_dir))

        colored_string("Testing outputs...", color="green")

        self.model_torch.eval()
        self.model_flow.eval()

        # Set inputs
        input_ids = self.get_random_tensor(0, 2000, (bs, seq_len), if_int=True)
        attention_mask = self.get_random_tensor(0, 2, sz=(bs, seq_len))
        token_type_ids = self.get_random_tensor(0, self.config.type_vocab_size, (bs, seq_len), if_int=True)
        position_ids = self.get_random_tensor(
            1, self.config.max_position_embeddings - 1, (bs, seq_len), if_int=True)
        head_mask = self.get_random_tensor(
            0, 2, (self.config.num_hidden_layers, self.config.num_attention_heads))
        labels = self.get_random_tensor(0, self.config.num_labels, (bs, ))
        inputs_embeds = None
        output_attentions = True
        output_hidden_states = True

        # Run forward
        colored_string("Running model with oneflow...", color="green", end="")
        out_flow = self.model_flow(input_ids[0], attention_mask[0], token_type_ids[0], position_ids[0], head_mask[0], 
                            inputs_embeds, labels[0], output_attentions, output_hidden_states)
        loss_flow, scores_flow, pkv_flow, hidden_flow, attn_flow, cross_attn_flow = out_flow
        colored_string("Done.", color="green")
        colored_string("Running model with transformers...", color="green", end="")
        output = self.model_torch(input_ids[1], attention_mask[1], token_type_ids[1], position_ids[1], head_mask[1], 
                            inputs_embeds, labels[1], output_attentions, output_hidden_states, return_dict=True)
        loss_torch = output.loss
        scores_torch = output.logits
        colored_string("Done.", color="green")

        # Calculate errors
        colored_string("Calculating errors...", color="green")
        loss_error = self.L1Loss_numpy(loss_flow, loss_torch)
        scores_error = self.L1Loss_numpy(scores_flow, scores_torch)
        colored_string("Loss error:{}".format(
            loss_error.item()), color="green")
        colored_string("Logits error:{}".format(
            scores_error.item()), color="green")
        colored_string("Done.", color="green")