import oneflow as flow

def gelu_new(x):
    gelu = flow.nn.GELU(approximate="tanh")
    return gelu(x)

ACT2FN = {
    "relu": flow.nn.functional.relu,
    "gelu": flow.nn.functional.gelu,
    "tanh": flow.nn.functional.tanh,
    "gelu_new": gelu_new,
    "sigmoid": flow.nn.functional.sigmoid,
}

