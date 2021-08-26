# alexnet
from model.flow_model.alexnet import alexnet as alexnet_of
from model.torch_model.alexnet import alexnet as alexnet_torch



def build_model(args):
    if args.eval_mode == "flow":
        if args.model == "alexnet":
            return {"model": alexnet_of(), "weight":"/data/rentianhe/code/new_models/models/eval_acc/weight/torch/alexnet-owt-4df8aa71.pth"}
        print("successfully build oneflow model")
    
    if args.eval_mode == "torch":
        if args.model == "alexnet":
            return {"model": alexnet_torch(), "weight":"/data/rentianhe/code/new_models/models/eval_acc/weight/torch/alexnet-owt-4df8aa71.pth"}
        print("successfully build pytorch model")


if __name__ == "__main__":
    inputs = torch.randn(10, 3, 224, 224)
    net = alexnet_torch()
    output = net(inputs)
    print(output.detach().numpy())