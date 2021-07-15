import oneflow.experimental as flow


def load_checkpoint(
    model, path_to_checkpoint,
):
    """
    Load the checkpoint from the given file. If inflation is True, inflate the
    2D Conv weights from the checkpoint to 3D Conv.
    Args:
        path_to_checkpoint (string): path to the checkpoint to load.
        model (model): model to load the weights from the checkpoint.
    """
    checkpoint = flow.load(path_to_checkpoint)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "No state_dict found in checkpoint file {}".format(path_to_checkpoint)
        )

    # get state_dict from checkpoint
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    # parameters=model.state_dict()
    # print(parameters)
    # for key, value in parameters.items():
    #     print(value)

    return model
