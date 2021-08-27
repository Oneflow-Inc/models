# alexnet
from model.flow_model.alexnet import alexnet as alexnet_of
from model.torch_model.alexnet import alexnet as alexnet_torch



def build_model(args):
    if args.eval_mode == "flow":
        if args.model == "alexnet":
            print("successfully build oneflow alexnet model")
            return alexnet_of()
    
    if args.eval_mode == "torch":
        if args.model == "alexnet":
            print("successfully build pytorch alexnet model")
            return alexnet_torch()


