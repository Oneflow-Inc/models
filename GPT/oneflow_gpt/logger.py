import oneflow as flow


def print_rank_0(*args, **kwargs):
    if flow.env.get_rank() == 0:
        print(*args, **kwargs)


def print_rank_last(*args, **kwargs):
    if flow.env.get_rank() == flow.env.get_world_size() - 1:
        print(*args, **kwargs)
