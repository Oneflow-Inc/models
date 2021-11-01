import oneflow as flow


def cc(net):
    device = flow.device("cuda")
    return net.to(device)


def infinite_iter(iterable):
    it = iter(iterable)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(iterable)
