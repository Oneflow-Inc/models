import json
import oneflow as flow
import oneflow.nn as nn

from models.roberta import Roberta


class SST2RoBERTa(nn.Module):
    def __init__(self, args, pretrain_dir, kwargs_path, hidden_size, num_labels,is_train=False):
        super(SST2RoBERTa, self).__init__()
        # with open(kwargs_path, "r") as f:
        #     kwargs = json.load(f)
        # model = Roberta(**kwargs)
        model = Roberta(args)
        if is_train == True:
            model.load_state_dict(flow.load(pretrain_dir))
        self.roberta = model
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, masks):
        outputs = self.roberta(inputs, masks)
        outputs = outputs[0][:, 0, :]
        outputs = self.classifier(outputs)
        outputs = self.softmax(outputs)
        return outputs
