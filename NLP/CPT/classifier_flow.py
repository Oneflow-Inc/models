import json
import os

import oneflow as flow
import oneflow.nn as nn

from models.CPT import CPT


class ClueAFQMCCPT(nn.Module):
    def __init__(self, pretrain_dir, num_labels, is_train):
        super(ClueAFQMCCPT, self).__init__()
        kwargs_path = os.path.join(pretrain_dir, "parameters.json")
        with open(kwargs_path, "r") as f:
            kwargs = json.load(f)
        model = CPT(**kwargs)
        if is_train == True:
            model.load_state_dict(flow.load(os.path.join(pretrain_dir, "weights")))
        self.cpt = model
        self.classifier = nn.Linear(model.d_model, num_labels)

    def forward(self, inputs, masks):
        outputs = self.cpt(inputs, masks)
        outputs = outputs[0][:, 0, :]
        outputs = self.classifier(outputs)
        return outputs
