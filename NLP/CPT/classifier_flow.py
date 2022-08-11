import json
import os

import oneflow as flow
import oneflow.nn as nn

from models.CPT import CPT


class ClueAFQMCCPT(nn.Module):
    def __init__(self, pretrain_dir, num_labels, is_train):
        super(ClueAFQMCCPT, self).__init__()
        kwargs_path = os.path.join(pretrain_dir, "parameters.json")
        self.cpt = self.load_model(pretrain_dir, kwargs_path, is_train)
        self.classifier = nn.Linear(self.cpt.d_model, num_labels)

    def forward(self, inputs, masks):
        outputs = self.cpt(inputs, masks)
        outputs = outputs[0][:, 0, :]
        outputs = self.classifier(outputs)
        return outputs

    def load_model(self, pretrain_dir, kwargs_path, is_train):
        with open(kwargs_path, "r") as f:
            kwargs = json.load(f)
        model = CPT(**kwargs)
        if is_train == True:
            model.load_state_dict(flow.load(os.path.join(pretrain_dir, "weights")))
        return model