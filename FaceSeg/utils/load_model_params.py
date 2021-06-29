# -*- coding:utf-8 -*-
import torch
import time


def load_model_params(model,model_path):

    start_t = time.time()

    dic = model.state_dict()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    start_t = time.time()
    torch_params = torch.load(model_path)
    torch_keys = torch_params.keys()

    for k in dic.keys():
        if k in torch_keys:
            dic[k] = torch_params[k].cpu().detach().numpy()

    model.load_state_dict(dic)
    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))
    return model