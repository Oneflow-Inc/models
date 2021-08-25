import numpy as np

import time
import tempfile
import os
import importlib.util
import argparse
from typing import Sequence
import subprocess
import re


import oneflow as flow
import oneflow._oneflow_internal as oneflow_internal


DEFAULT_TIMES = 20
gpu_memory_used_by_pytorch = 0


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


def gpu_memory_used():
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'])
    output = output.decode('utf-8').strip()
    my_pid = os.getpid()
    mem_used_by_me = 0
    for line in output.split('\n'):
        pid, mem_used = map(int, re.split(",? ", line)[:2])
        if pid == my_pid:
            mem_used_by_me += mem_used
    return mem_used_by_me


def print_rank_0(*args, **kwargs):
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        print(*args, **kwargs)


def test(
    model_path: str,
    module_name: str,
    input_shape: Sequence[int],
    disable_backward=False,
    times=DEFAULT_TIMES,
    no_verbose=False,
    ddp=False,
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

        if ddp:
            import torch.distributed as dist

            local_rank_env_var = os.getenv("LOCAL_RANK")
            assert local_rank_env_var is not None
            rank = int(local_rank_env_var)
            torch.cuda.set_device(rank)

            dist.init_process_group(backend="nccl", init_method="env://")

    Net = getattr(python_module, module_name)

    warmup_times = 5

    m = Net()
    m = m.to("cuda")

    if ddp:
        if test_oneflow:
            m = torch.nn.parallel.DistributedDataParallel(m)
        else:
            m = torch.nn.parallel.DistributedDataParallel(m, device_ids=[rank])

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
    total_time_ms = (end - start) * 1000
    time_per_run_ms = total_time_ms / times
    if no_verbose:
        print_rank_0(f"{framework_name}: {time_per_run_ms:.1f}ms")
    else:
        print_rank_0(
            f"{framework_name} {module_name} time: {time_per_run_ms:.1f}ms (= {total_time_ms:.1f}ms / {times}, input_shape={input_shape}, backward is {'disabled' if disable_backward else 'enabled'})"
        )
    global gpu_memory_used_by_pytorch
    if test_oneflow:
        print_rank_0(f"{framework_name} GPU used (rank 0): {gpu_memory_used() - gpu_memory_used_by_pytorch} MiB")
    else:
        gpu_memory_used_by_pytorch = gpu_memory_used()

        print_rank_0(f"{framework_name} GPU used (rank 0): {gpu_memory_used_by_pytorch} MiB")
    if ddp and not test_oneflow:
        import torch.distributed as dist

        dist.destroy_process_group()

    return time_per_run_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("module_name", type=str)
    parser.add_argument("input_shape", type=str)
    parser.add_argument("--times", type=int, default=DEFAULT_TIMES)
    parser.add_argument("--disable-backward", action="store_true")
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    args = parser.parse_args()
    input_shape = list(map(int, args.input_shape.split("x")))

    global test_oneflow

    # NOTE: PyTorch must run before OneFlow for correct memory usage
    test_oneflow = False
    pytorch_time = test(
        args.model_path,
        args.module_name,
        input_shape,
        disable_backward=args.disable_backward,
        times=args.times,
        no_verbose=args.no_verbose,
        ddp=args.ddp,
    )

    test_oneflow = True
    oneflow_time = test(
        args.model_path,
        args.module_name,
        input_shape,
        disable_backward=args.disable_backward,
        times=args.times,
        no_verbose=args.no_verbose,
        ddp=args.ddp,
    )
    relative_speed = pytorch_time / oneflow_time
    if args.no_verbose:
        print_rank_0(f"Relative speed: {relative_speed:.2f}")
    else:
        print_rank_0(
            f"Relative speed: {relative_speed:.2f} (= {pytorch_time:.1f}ms / {oneflow_time:.1f}ms)"
        )
