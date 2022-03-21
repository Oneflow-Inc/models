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
import argparse
import oneflow as flow


def get_args(print_args=True):
    def int_list(x):
        return list(map(int, x.split(",")))

    def str_list(x):
        return list(map(str, x.split(",")))

    parser = argparse.ArgumentParser()

    parser.add_argument("--use_fusedmlp", action="store_true", help="use fused MLP or not")
    parser.add_argument("--embedding_vec_size", type=int, default=128)
    parser.add_argument("--bottom_mlp", type=int_list, default="512,256,128")
    parser.add_argument("--top_mlp", type=int_list, default="1024,1024,512,256")
    parser.add_argument(
        "--disable_interaction_padding",
        action="store_true",
        help="disable interaction padding or not",
    )
    parser.add_argument(
        "--interaction_itself", action="store_true", help="interaction itself or not"
    )
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_save_dir", type=str, default=None)
    parser.add_argument(
        "--save_initial_model", action="store_true", help="save initial model parameters or not.",
    )
    parser.add_argument(
        "--save_model_after_each_eval", action="store_true", help="save model after each eval.",
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--eval_batches", type=int, default=1612, help="number of eval batches")
    parser.add_argument("--eval_batch_size", type=int, default=55296)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--train_batch_size", type=int, default=55296)
    parser.add_argument("--learning_rate", type=float, default=24)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=27772)
    parser.add_argument("--decay_start", type=int, default=49315)
    parser.add_argument("--max_iter", type=int, default=75000)
    parser.add_argument("--loss_print_interval", type=int, default=1000)
    parser.add_argument(
        "--column_size_array", type=int_list, help="column_size_array", required=True
    )
    parser.add_argument(
        "--persistent_path", type=str, required=True, help="path for persistent kv store",
    )
    parser.add_argument("--store_type", type=str, default="device_host")
    parser.add_argument("--cache_memory_budget_mb_per_rank", type=int, default=8192)
    parser.add_argument("--amp", action="store_true", help="Run model with amp")
    parser.add_argument("--loss_scale_policy", type=str, default="static", help="static or dynamic")

    args = parser.parse_args()

    if print_args and flow.env.get_rank() == 0:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


if __name__ == "__main__":
    get_args()
