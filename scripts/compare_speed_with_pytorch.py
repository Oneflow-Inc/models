import numpy as np

import time
import tempfile
import os
import importlib.util
import argparse
from typing import Sequence


import oneflow.experimental as flow
import oneflow._oneflow_internal as oneflow_internal

flow.enable_eager_execution()

DEFAULT_TIMES = 20


def import_file(path):
    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sync(x):
    if test_oneflow:
        x.numpy()
    else:
        x.cpu()


def test(
    model_path: str,
    module_name: str,
    input_shape: Sequence[int],
    disable_backward=False,
    times=DEFAULT_TIMES,
    time_only=False,
):
    framework_name = "OneFlow" if test_oneflow else "PyTorch"
    if test_oneflow:
        python_module = import_file(model_path)
        torch = flow
    else:
        with open(model_path) as f:
            buf = f.read()

        lines = buf.split("\n")
        for i, line in enumerate(lines):
            if "import" not in line and len(line.strip()) != 0:
                break
        lines = (
            lines[:i]
            + [
                "import torch as flow",
                "import torch.nn as nn",
                "from torch import Tensor",
                "from torch.nn import Parameter",
            ]
            + lines[i:]
        )
        buf = "\n".join(lines)
        with tempfile.NamedTemporaryFile("w", suffix=".py") as f:
            f.write(buf)
            python_module = import_file(f.name)

        import torch
    Net = getattr(python_module, module_name)

    times = 20
    warmup_times = 5

    m = Net()
    m = m.to("cuda")

    def run_model(m, x):
        if disable_backward:
            with torch.no_grad():
                return m(x)
        else:
            return m(x)

    learning_rate = 0.01
    mom = 0.9
    optimizer = torch.optim.SGD(m.parameters(), lr=learning_rate, momentum=mom)

    # input tensor of OneFlow should set requires_grad=False due to a bug
    x = torch.tensor(
        np.ones(input_shape).astype(np.float32), requires_grad=not test_oneflow
    ).to("cuda")
    for i in range(warmup_times + times):
        if i == warmup_times:
            start = time.time()
        y = run_model(m, x)
        if not disable_backward:
            y = y.sum()
            y.backward()
            optimizer.zero_grad()
            optimizer.step()
        sync(y)
    end = time.time()
    total_time = end - start
    time_per_run = total_time / times
    if time_only:
        print(time_per_run)
    else:
        print(
            f"{framework_name} {module_name} time: {time_per_run} (= {total_time} / {times}, input_shape={input_shape}, backward is {'disabled' if disable_backward else 'enabled'})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("module_name", type=str)
    parser.add_argument("input_shape", type=str)
    parser.add_argument("--times", type=int, default=DEFAULT_TIMES)
    parser.add_argument("--disable-backward", action="store_true")
    parser.add_argument("--time-only", action="store_true")

    args = parser.parse_args()
    input_shape = list(map(int, args.input_shape.split("x")))

    global test_oneflow

    test_oneflow = False
    test(
        args.model_path,
        args.module_name,
        input_shape,
        disable_backward=args.disable_backward,
        times=args.times,
        time_only=args.time_only,
    )

    test_oneflow = True
    test(
        args.model_path,
        args.module_name,
        input_shape,
        disable_backward=args.disable_backward,
        times=args.times,
        time_only=args.time_only,
    )
