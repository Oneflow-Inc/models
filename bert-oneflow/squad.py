"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import oneflow as flow
import oneflow.nn as nn
from modeling import BertModel

class SQuAD(nn.Module):
    def __init__(self, vocab_size,
        seq_length,
        hidden_size,
        hidden_layers,
        atten_heads,
        intermediate_size,
        hidden_act,
        hidden_dropout_prob,
        attention_probs_dropout_prob,
        max_position_embeddings,
        type_vocab_size,
        initializer_range=0.02):
        super().__init__()
        self.bert = BertModel(
            vocab_size,
            seq_length,
            hidden_size,
            hidden_layers,
            atten_heads,
            intermediate_size,
            hidden_act,
            hidden_dropout_prob,
            attention_probs_dropout_prob,
            max_position_embeddings,
            type_vocab_size,
        )
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.cls_squad = nn.Linear(hidden_size, 2)

        self.cls_squad.weight.data.normal_(mean=0., std=initializer_range)
        self.cls_squad.bias.data.fill_(0)

    def forward(self, input_ids, token_type_ids, attention_mask):
        sequence_output, _ = self.bert(
            input_ids, token_type_ids, attention_mask
        )
        final_hidden = flow.reshape(sequence_output, [-1, self.hidden_size])

        prediction_logits = self.cls_squad(final_hidden)
        prediction_logits = flow.reshape(prediction_logits, [-1, self.seq_length, 2])
        start_logits = prediction_logits[:, :, 0]
        end_loigts = prediction_logits[:, :, 1]
        return start_logits, end_loigts
