import oneflow as flow


def ttol(tensor, pure_local=True):
    """ to local """
    if tensor.is_consistent:
        if pure_local:
            tensor = tensor.to_local()
        else:
            tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local()

    return tensor


def tton(tensor, local_only=True):
    """ tensor to numpy """
    if tensor.is_consistent:
        if local_only:
            tensor = tensor.to_local().numpy()
        else:
            tensor = tensor.to_consistent(sbp=flow.sbp.broadcast).to_local().numpy()
    else:
        tensor = tensor.numpy()

    return tensor
