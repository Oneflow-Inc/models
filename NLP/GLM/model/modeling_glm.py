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

"""GPT-2 model."""

import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F

import mpu
from utils import print_rank_0


def init_method_normal(std=0.02):
    def init_(tensor):
        return flow.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class GLMModel(flow.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0,
                 ):

        super(GLMModel, self).__init__()
        
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.embedding_dropout_prob = embedding_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.output_dropout_prob = output_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.max_memory_length = max_memory_length
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.parallel_output = parallel_output
        self.relative_encoding = relative_encoding
        self.block_position_encoding = block_position_encoding
        self.output_predict = output_predict
        self.spell_length = spell_length
        self.spell_func = spell_func
        self.attention_scale = attention_scale


        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       relative_encoding=relative_encoding,
                                                       block_position_encoding=block_position_encoding)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        batch_size = input_ids.size(0)
        
        flow._oneflow_internal.profiler.RangePush('embeddings')
        words_embeddings = self.word_embeddings(input_ids)
        flow._oneflow_internal.profiler.RangePop()
        embeddings = words_embeddings
        #False
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = flow._C.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        flow._oneflow_internal.profiler.RangePush('transformer')
        transformer_output = self.transformer(embeddings, position_ids, attention_mask)
        flow._oneflow_internal.profiler.RangePop()
        logits, hidden_layers = transformer_output
        outputs = hidden_layers
        
        #True
        if self.output_predict:

            # logits_parallel = mpu.copy_to_model_parallel_region(
            #     logits)
            logits_parallel = logits
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)
            
            if self.parallel_output:
                return (logits_parallel, *outputs)
            
            return (logits_parallel,*outputs)
        else:
            return (logits, *outputs)
    
    @classmethod
    def load_model(cls, path, cuda=True):
        package = flow.load(path)
        model = cls(num_layers = package['num_layers'],
                    vocab_size =  package['vocab_size'],
                    hidden_size =  package['hidden_size'],
                    num_attention_heads =  package['num_attention_heads'],
                    embedding_dropout_prob =  package['embedding_dropout_prob'],
                    attention_dropout_prob =  package['attention_dropout_prob'],
                    output_dropout_prob =  package['output_dropout_prob'],
                    max_sequence_length =  package['max_sequence_length'],
                    max_memory_length =  package['max_memory_length'],
                    checkpoint_activations =  package['checkpoint_activations'],
                    checkpoint_num_layers =  package['checkpoint_num_layers'],
                    parallel_output =  package['parallel_output'],
                    relative_encoding =  package['relative_encoding'],
                    block_position_encoding =  package['block_position_encoding'],
                    output_predict =  package['output_predict'],
                    spell_length =  package['spell_length'],
                    spell_func =  package['spell_func'],
                    attention_scale =  package['attention_scale']
                   )
        model.load_state_dict(package['state_dict'])
        if cuda:
            model.cuda()
        return model

    @staticmethod
    def serialize(model):
        package = {
        "num_layers" : model.num_layers,
        "vocab_size" : model.vocab_size,
        "hidden_size" : model.hidden_size,
        "num_attention_heads" : model.num_attention_heads,
        "embedding_dropout_prob" : model.embedding_dropout_prob,
        "attention_dropout_prob" : model.attention_dropout_prob,
        "output_dropout_prob" : model.output_dropout_prob,
        "max_sequence_length" : model.max_sequence_length,
        "max_memory_length" : model.max_memory_length,
        "checkpoint_activations" : model.checkpoint_activations,
        "checkpoint_num_layers" : model.checkpoint_num_layers,
        "parallel_output" : model.parallel_output,
        "relative_encoding" : model.relative_encoding,
        "block_position_encoding" : model.block_position_encoding,
        "output_predict" : model.output_predict,
        "spell_length" : model.spell_length,
        "spell_func" : model.spell_func,
        "attention_scale" : model.attention_scale,
        'state_dict': model.state_dict()
        }

        # package = {
        #     'vocab_size': model.vocab_size,
        #     'embedding_size': model.embedding_size,
        #     'hidden_size': model.hidden_size,
        #     'num_layers': model.num_layers,
        #     'num_class': model.num_class,
        #     'state_dict': model.state_dict(),
        #     'optim_dict': optimizer.state_dict(),
        #     'epoch': epoch
        #     }
        return package

class EncoderDecoder(flow.nn.Module):

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 output_predict=True
                 ):
        super(EncoderDecoder, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.encoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers)
        self.decoder = mpu.GPT2ParallelTransformer(num_layers,
                                                   hidden_size,
                                                   num_attention_heads,
                                                   max_sequence_length,
                                                   max_memory_length,
                                                   embedding_dropout_prob,
                                                   attention_dropout_prob,
                                                   output_dropout_prob,
                                                   checkpoint_activations,
                                                   checkpoint_num_layers,
                                                   use_decoder_layer=True)

    def forward(self, source_ids, target_ids, source_position_ids, target_position_ids, source_mask, target_mask):
        # Embeddings.
        source_embeddings = self.word_embeddings(source_ids)
        target_embeddings = self.word_embeddings(target_ids)

        # Transformer.
        encoder_output, _ = self.encoder(source_embeddings, source_position_ids, source_mask)
        decoder_output, _ = self.decoder(target_embeddings, target_position_ids, target_mask)
        if self.output_predict:
            # Parallel logits.
            output_parallel = mpu.copy_to_model_parallel_region(decoder_output)
            logits_parallel = F.linear(output_parallel, self.word_embeddings.weight)
            return (logits_parallel,)
        else:
            return (decoder_output,)


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        #False
        if isinstance(module_, (mpu.LayerNorm, flow.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])
    return weight_decay_params, no_weight_decay_params
