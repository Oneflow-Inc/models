
import oneflow.nn as nn
import oneflow as flow
from modeling import BertForPreTraining


def change_name_from_lazy_to_eager(lazy_name: str):
    eager_name = lazy_name.replace('-', '.').replace('layer_', 'layer.')
    # In lazy model, params in layernorm are named as `gamma` and `beta`, 
    # but in eager model, these params are named as `weight` and `bias`
    if "LayerNorm" in eager_name:
        eager_name = eager_name.replace("gamma", "weight").replace("beta", "bias")
    return eager_name



if __name__ == "__main__":
    lazy_state_dict = flow.load("../../OneFlow-Benchmark/LanguageModeling/BERT/snapshots/snapshot_snapshot_1")

    bert_module = BertForPreTraining(
        30522,
        128,
        768,
        12,
        12,
        3072,
        nn.GELU(),
        0.1,
        0.1,
        512,
        2,
    )

    all_eager_names_list = set(bert_module.state_dict().keys())

    for lazy_name, lazy_weight in lazy_state_dict.items():
        # skip momentum and momentum^2 for optimizer
        if lazy_name.endswith("-v") or lazy_name.endswith("-m"):
            continue
        eager_name = change_name_from_lazy_to_eager(lazy_name)
        if eager_name not in all_eager_names_list:
            print(f"{eager_name} is not matched")
            continue
        else:
            all_eager_names_list.remove(eager_name)
            bert_module.state_dict()[eager_name].data.copy_(lazy_weight)
    

    """
    num_valid_lazy = 0
    for lazy_name, lazy_weight in lazy_state_dict.items():
        if lazy_name.endswith("-v") or lazy_name.endswith("-m"):
            continue
        eager_name = change_name_from_lazy_to_eager(lazy_name)
        if eager_name not in all_eager_names_list:
            print(f"{eager_name} is not matched")
        else:
            num_valid_lazy += 1
            all_eager_names_list.remove(eager_name)

    print(f"number of lazy params is {num_valid_lazy}\n"
          f"number of eager params is {len(all_eager_names_list)}")
    print(all_eager_names_list)
    # eager_state_dict = bert_module.state_dict()

    # lazy_name = list(lazy_state_dict.keys())[5]
    # eager_name = list(eager_state_dict.keys())[6]

    # print(lazy_name)
    # print(eager_name)

    # lazy_weight = lazy_state_dict[lazy_name]
    # eager_weight = eager_state_dict[eager_name]
    """

