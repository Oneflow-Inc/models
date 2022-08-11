import json
import oneflow as flow
import oneflow.nn as nn

from models.roberta import Roberta


class SST2RoBERTa(nn.Module):
    def __init__(self, pretrain_dir, kwargs_path, hidden_size, num_labels, is_train):
        super(SST2RoBERTa, self).__init__()
        self.roberta = self.load_model(pretrain_dir, kwargs_path, is_train)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, masks):
        outputs = self.roberta(inputs, masks)
        outputs = outputs[0][:, 0, :]
        outputs = self.classifier(outputs)
        outputs = self.softmax(outputs)
        return outputs

    def load_model(self, pretrain_dir, kwargs_path, is_train):
        with open(kwargs_path, "r") as f:
            kwargs = json.load(f)
        model = Roberta(**kwargs)
        if is_train == True:
            model.load_state_dict(flow.load(pretrain_dir))
        return model
