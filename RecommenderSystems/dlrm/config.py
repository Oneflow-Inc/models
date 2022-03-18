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

    parser.add_argument("--mlp_type", type=str, default="FusedMLP", help="MLP or FusedMLP")
    parser.add_argument("--bottom_mlp", type=int_list, default="512,256,128")
    parser.add_argument("--top_mlp", type=int_list, default="1024,1024,512,256")
    parser.add_argument('--output_padding', type=int, default=1,
                        help="interaction output padding size")
    parser.add_argument("--interaction_itself", action="store_true",
                        help="interaction itself or not")
    parser.add_argument("--model_load_dir", type=str, default="")
    parser.add_argument("--model_save_dir", type=str, default="./checkpoint")
    parser.add_argument("--save_initial_model", action="store_true",
                        help="save initial model parameters or not.")
    parser.add_argument("--save_model_after_each_eval", action="store_true",
                        help="save model after each eval.")
    parser.add_argument("--eval_after_training", action="store_true",
                        help="do eval after_training")
    parser.add_argument("--data_dir", type=str, default="/dataset/dlrm_parquet")
    parser.add_argument('--eval_batchs', type=int, default=-1, 
                        help="<0: whole val ds, 0: do not val, >0: number of eval batches")
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument("--eval_batch_size_per_proc", type=int, default=None)
    parser.add_argument('--eval_interval', type=int, default=1000)    
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--batch_size_per_proc", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_batches", type=int, default=2750)
    parser.add_argument("--decay_batches", type=int, default=27772)
    parser.add_argument("--decay_start", type=int, default=49315)
    parser.add_argument("--embedding_vec_size", type=int, default=128)
    parser.add_argument("--num_dense_fields", type=int, default=13)
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--loss_print_every_n_iter", type=int, default=100)
    parser.add_argument("--num_sparse_fields", type=int, default=26)
    parser.add_argument("--column_size_array", type=int_list, help="column_size_array")
    parser.add_argument("--persistent_path", type=str, default="", 
                        help="path for persistent kv store")
    parser.add_argument("--cache_type", type=str, default="device_host")
    parser.add_argument("--cache_memory_budget_mb", type=int_list, default="8192,16384")
    parser.add_argument("--use_fp16", action="store_true", help="Run model with amp")
    parser.add_argument("--loss_scale_policy", type=str, default="static", 
                        help="static or dynamic")

    args = parser.parse_args()

    world_size = flow.env.get_world_size()
    if args.batch_size_per_proc is None:
        assert args.batch_size % world_size == 0
        args.batch_size_per_proc = args.batch_size // world_size
    elif args.batch_size is None:
        args.batch_size = args.batch_size_per_proc * world_size
    else:
        assert args.batch_size % args.batch_size_per_proc == 0

    if args.eval_batch_size_per_proc is None:
        assert args.eval_batch_size % world_size == 0
        args.eval_batch_size_per_proc = args.eval_batch_size // world_size
    elif args.eval_batch_size is None:
        args.eval_batch_size = args.eval_batch_size_per_proc * world_size
    else:
        assert args.eval_batch_size % args.eval_batch_size_per_proc == 0

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
