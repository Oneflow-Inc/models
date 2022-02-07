import oneflow as flow
import oneflow.nn as nn


def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )


def meter(self, mkey, *args):
    assert mkey in self.m
    self.m[mkey]["meter"].record(*args)


class EvalGraph(flow.nn.Graph):
    def __init__(self, model,fp16=False):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)
        self.model = model
        if fp16:
            self.config.enable_amp(True)

    def build(self, image):
        logits = self.model(image)
        return logits
