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

import unittest
from collections import OrderedDict
import tempfile

import numpy as np
from test_util import GenArgDict
#from optimizer_test_util import clip_grad_norm_np

import oneflow as flow
from oneflow.nn.parameter import Parameter

import torch 


def clip_grad_norm_np(np_grad, max_norm, norm_type):
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == float("inf"):
        total_norm = np.max(np.abs(np_grad))
    if norm_type == float("-inf"):
        total_norm = np.min(np.abs(np_grad))
    elif norm_type == 0:
        total_norm = np.sum(np.stack([np.sum(np_grad != 0)]) != 0)
    else:
        total_norm = np_grad
        for i in range(np_grad.ndim, 0, -1):
            total_norm = np.linalg.norm(total_norm, norm_type, axis=i - 1)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        np_grad = np_grad * clip_coef
    return total_norm, np_grad





class TrainGraph(flow.nn.Graph):
    def __init__(self, model,optimizer):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)

        self.model = model
        self.add_optimizer(optimizer)
    def build(self,image):

        image = image
        logits = self.model(image)
        return logits

def compare_with_numpy_sgd(
    test_case,
    device,
    x_shape,
    momentum,
    weight_decay,
    learning_rate,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32)/10000.)
    init_value = np.random.uniform(size=x_shape).astype(np.float32)
    weight_decay=0
    momentum=0.9
    learning_rate=0.1
    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        sgd = flow.optim.SGD(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad,
                dtype=flow.float32,
                requires_grad=False,
                device=flow.device(device),
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD([x])
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
                        file_name = f.name
                        import pickle

                        pickle.dump(state_dict, f)
                    with open(file_name, "rb") as f:
                        state_dict = pickle.load(f)
                sgd.load_state_dict(state_dict)
        return x



    def train_by_torch():
        x = torch.nn.Parameter(torch.tensor(init_value, device=torch.device(device)))
        sgd = torch.optim.SGD(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                }
            ]
        )

        def train_one_iter(grad):
            
            grad_tensor = torch.tensor(
                grad,
                dtype=torch.float32,
                requires_grad=False,
                device=torch.device(device),
            )
            loss = torch.sum(x * grad_tensor)
            loss.backward()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = torch.optim.SGD(            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                }
            ])
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
                        file_name = f.name
                        import pickle

                        pickle.dump(state_dict, f)
                    with open(file_name, "rb") as f:
                        state_dict = pickle.load(f)
                sgd.load_state_dict(state_dict)
        return x


    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def train_one_iter(grad):
            grad = grad + weight_decay 
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)


        # def train_one_iter(grad):
        #     grad = grad + weight_decay
        #     v = momentum * vt + learning_rate * grad
        #     param = x- v
        #     return (param, v)




        for i in range(train_iters):
            (x, vt) = train_one_iter(random_grad_seq[i])
        return x
    
    
    torch_res = train_by_torch().cpu().detach().numpy()
    numpy_res = train_by_numpy()
    oneflow_res = train_by_oneflow().numpy()
    print("##########")
   #print(init_value)
    
    print(oneflow_res)
    print(numpy_res)
    print(torch_res)
    test_case.assertTrue(
        np.allclose(
            oneflow_res.flatten(), numpy_res.flatten(), rtol=0.0001, atol=0.0001
        )
    )


def compare_with_numpy_sgd_clip_grad(
    test_case,
    device,
    x_shape,
    momentum,
    weight_decay,
    learning_rate,
    clip_grad_max_norm,
    clip_grad_norm_type,
    train_iters,
    reload_state_step,
    save_load_by_pickle,
):
    random_grad_seq = []
    for _ in range(train_iters):
        random_grad_seq.append(np.random.uniform(size=x_shape).astype(np.float32))
    init_value = np.random.uniform(size=x_shape).astype(np.float32)

    def train_by_oneflow():
        x = Parameter(flow.Tensor(init_value, device=flow.device(device)))
        sgd = flow.optim.SGD(
            [
                {
                    "params": [x],
                    "lr": learning_rate,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "clip_grad_max_norm": clip_grad_max_norm,
                    "clip_grad_norm_type": clip_grad_norm_type,
                }
            ]
        )

        def train_one_iter(grad):
            grad_tensor = flow.tensor(
                grad,
                dtype=flow.float32,
                requires_grad=False,
                device=flow.device(device),
            )
            loss = flow.sum(x * grad_tensor)
            loss.backward()
            sgd.clip_grad()
            sgd.step()
            sgd.zero_grad()

        for i in range(train_iters):
            train_one_iter(random_grad_seq[i])
            # test state_dict/load_state_dict
            if i == reload_state_step:
                state_dict = sgd.state_dict()
                sgd = flow.optim.SGD([x])
                if save_load_by_pickle:
                    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
                        file_name = f.name
                        import pickle

                        pickle.dump(state_dict, f)
                    with open(file_name, "rb") as f:
                        state_dict = pickle.load(f)
                sgd.load_state_dict(state_dict)
        return x

    def train_by_numpy():
        x = init_value
        vt = np.zeros_like(x)

        def train_one_iter(grad):
            total_norm, grad = clip_grad_norm_np(
                grad, clip_grad_max_norm, clip_grad_norm_type
            )
            grad = grad + weight_decay * x
            v = momentum * vt - learning_rate * grad
            param = x + v
            return (param, v)

        for i in range(train_iters):
            (x, vt) = train_one_iter(random_grad_seq[i])
        return x

    oneflow_res = train_by_oneflow().numpy()
    numpy_res = train_by_numpy()

    test_case.assertTrue(
        np.allclose(
            oneflow_res.flatten(), numpy_res.flatten(), rtol=0.0001, atol=0.0001
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestOptimizers(flow.unittest.TestCase):
    def test_sgd(test_case):
        arg_dict = OrderedDict()
        arg_dict["device"] = [ "cuda"]
        arg_dict["x_shape"] = [(5,)]
        arg_dict["momentum"] = [ 0.9]
        arg_dict["weight_decay"] = [ 0.0005]
        arg_dict["learning_rate"] = [ 0.1]
        arg_dict["train_iters"] = [10]
        arg_dict["reload_state_step"] = [5]  # save and load optim state
        arg_dict["save_load_by_pickle"] = [ True]
        for arg in GenArgDict(arg_dict):
            compare_with_numpy_sgd(test_case, **arg)


if __name__ == "__main__":
    unittest.main()
