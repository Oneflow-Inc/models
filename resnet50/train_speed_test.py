import oneflow.experimental as flow
from oneflow.python.framework.function_util import global_function_or_identity

import numpy as np
import time
import argparse
import torch

import models.pytorch_resnet50 as pytorch_resnet50
from models.resnet50 import resnet50

def _parse_args():
    parser = argparse.ArgumentParser("flags for save style transform model")
    return parser.parse_args()

def rmse(l, r):
    return np.sqrt(np.mean(np.square(l - r)))

def main(args):
    flow.env.init()
    flow.enable_eager_execution()
    batch_size = 1
    image_nd = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    label_nd = np.array([e for e in range(batch_size)], dtype=np.int32)

    start_t = time.time()
    res50_module = resnet50()
    dic = res50_module.state_dict()
    end_t = time.time()
    print('init time : {}'.format(end_t - start_t))

    start_t = time.time()
    torch_params = {}
    for k in dic.keys():
        torch_params[k] = torch.from_numpy(dic[k].numpy()) 

    end_t = time.time()
    print('load params time : {}'.format(end_t - start_t))

    # # set for eval mode
    res50_module.eval()
    start_t = time.time()

    image = flow.tensor(image_nd, requires_grad=True)
    label = flow.tensor(label_nd, dtype=flow.long, requires_grad=False).to('cuda')
    corss_entropy = flow.nn.CrossEntropyLoss(reduction="mean")

    image_gpu = image.to('cuda')
    label = label.to('cuda')
    res50_module.to('cuda')
    corss_entropy.to('cuda')

    learning_rate = 0.01
    mom = 0.9
    of_sgd = flow.optim.SGD(res50_module.parameters(), lr=learning_rate, momentum=mom)

    bp_iters = 10
    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    for i in range(bp_iters):
        s_t = time.time()
        logits = res50_module(image_gpu)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t
        
        s_t = time.time()
        of_sgd.step()
        of_sgd.zero_grad()
        update_time += time.time() - s_t

    of_loss = loss.numpy()
    of_in_grad = image.grad.numpy()
    predictions = logits.softmax()
    of_predictions = predictions.numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    print('fp time : {}'.format(for_time / bp_iters))
    print('bp time : {}'.format(bp_time / bp_iters))
    print('update time : {}'.format(update_time / bp_iters))

    #####################################################################################################
    # pytorch resnet50
    torch_res50_module = pytorch_resnet50.resnet50()
    start_t = time.time()
    print(type(start_t))
    torch_res50_module.load_state_dict(torch_params)
    end_t = time.time()
    print('torch load params time : {}'.format(end_t - start_t))

    # set for eval mode
    torch_res50_module.eval()
    torch_res50_module.to('cuda')

    torch_sgd = torch.optim.SGD(torch_res50_module.parameters(), lr=learning_rate, momentum=mom)

    start_t = time.time()
    image = torch.tensor(image_nd, requires_grad=True)
    image_gpu = image.to('cuda')
    corss_entropy = torch.nn.CrossEntropyLoss()
    corss_entropy.to('cuda')
    label = torch.tensor(label_nd, dtype=torch.long, requires_grad=False).to('cuda')


    for_time = 0.0
    bp_time = 0.0
    update_time = 0.0

    for i in range(bp_iters):
        s_t = time.time()
        logits = torch_res50_module(image_gpu)
        loss = corss_entropy(logits, label)
        for_time += time.time() - s_t

        s_t = time.time()
        loss.backward()
        bp_time += time.time() - s_t

        s_t = time.time()
        torch_sgd.step()
        torch_sgd.zero_grad()
        update_time += time.time() - s_t
        
    torch_loss = loss.cpu().detach().numpy()
    torch_in_grad = image.grad.cpu().detach().numpy()
    predictions = logits.softmax(-1)
    torch_predictions = predictions.cpu().detach().numpy()
    end_t = time.time()
    print('infer time : {}'.format(end_t - start_t))
    print('fp time : {}'.format(for_time / bp_iters))
    print('bp time : {}'.format(bp_time / bp_iters))
    print('update time : {}'.format(update_time / bp_iters))

if __name__ == "__main__":
    args = _parse_args()
    main(args)








