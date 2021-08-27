import oneflow as flow

from config import get_args
from models.optimizer import make_grad_scaler


def make_train_graph(model, cross_entropy, data_loader, optimizer, lr_scheduler=None):
    return TrainGraph(model, cross_entropy, data_loader, optimizer, lr_scheduler)


def make_eval_graph(model, data_loader):
    return EvalGraph(model, data_loader)


class TrainGraph(flow.nn.Graph):
    def __init__(self, model, cross_entropy, data_loader, optimizer, lr_scheduler=None):
        super().__init__()
        args = get_args()

        if args.use_fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(make_grad_scaler(args))

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.model = model
        self.cross_entropy = cross_entropy
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self):
        image, label = self.data_loader()
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        loss = self.cross_entropy(logits, label)
        loss.backward()
        return loss


class EvalGraph(flow.nn.Graph):
    def __init__(self, model, data_loader):
        super().__init__()

        args = get_args()
        if args.use_fp16:
            self.config.enable_amp(True)

        self.config.allow_fuse_add_to_output(True)

        self.data_loader = data_loader
        self.model = model

    def build(self):
        image, label = self.data_loader()
        logits = self.model(image.to("cuda"))
        predictions = logits.softmax()
        return predictions, label
