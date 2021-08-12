import torch

checkpoint_path = "/data/rentianhe/code/ViT-OneFlow/models/ViT/weights/checkpoint/imagenet21k+imagenet2012_ViT-B_16.pth"


def convert_torch_to_oneflow(path):
    state_dict = load_checkpoint(path)
    new_parameters = dict()
    for key, value in state_dict.items():
     if "num_batches_tracked" not in key:
          val = value.detach().cpu().numpy()
          new_parameters[key] = val
    return new_parameters


def load_checkpoint(path):
    state_dict = torch.load(path)['state_dict']
    keys, values = zip(*list(state_dict.items()))
    state_dict = convert_torch_weight(keys, values)
    return state_dict

def convert_torch_weight(keys, values):
    """ Convert jax model parameters with pytorch model parameters """
    state_dict = {}
    for key, value in zip(keys, values):

        # convert name to torch names
        names = key.split('.')
        torch_names = names
        # torch_names = replace_names(names)
        torch_key = key

        # convert values to tensor and check shapes
        tensor_value = value
        # check shape
        num_dim = len(tensor_value.shape)

        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            # for normal weight, transpose it
            tensor_value = tensor_value
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['query', 'key', 'value']:
            feat_dim, num_heads, head_dim = tensor_value.shape
            # for multi head attention q/k/v weight
            dim, head, head_dim = tensor_value.shape
            tensor_value = tensor_value.view(dim, head * head_dim).t()
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['query', 'key', 'value']:
            # for multi head attention q/k/v bias
            tensor_value = tensor_value.view(-1)
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            # for multi head attention out weight
            head, head_dim, dim = tensor_value.shape
            tensor_value = tensor_value.view(head * head_dim, dim).t()
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value

        # print("{}: {}".format(torch_key, tensor_value.shape))
        state_dict[torch_key] = tensor_value
    return state_dict

# print(load_checkpoint(checkpoint_path))
if __name__ == "__main__":
    state_dict = load_checkpoint(checkpoint_path)
    for key in state_dict.keys():
        print(state_dict[key].shape)