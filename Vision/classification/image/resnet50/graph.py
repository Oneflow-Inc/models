import oneflow as flow

from config import get_args
from models.optimizer import make_grad_scaler
from models.optimizer import make_static_grad_scaler


def make_train_graph(
    model, cross_entropy, data_loader, optimizer, lr_scheduler=None, *args, **kwargs
):
    return TrainGraph(
        model, cross_entropy, data_loader, optimizer, lr_scheduler, *args, **kwargs
    )


def make_eval_graph(model, data_loader):
    return EvalGraph(model, data_loader)


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        cross_entropy,
        data_loader,
        optimizer,
        lr_scheduler=None,
        return_pred_and_label=True,
    ):
        super().__init__()
        args = get_args()
        self.return_pred_and_label = return_pred_and_label

        if args.use_fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(make_grad_scaler())
        elif args.scale_grad:
            self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        # Disable cudnn_conv_heuristic_search_algo will open dry-run.
        # Dry-run is better with single device, but has no effect with multiple device.
        self.config.enable_cudnn_conv_heuristic_search_algo(False)
        self.world_size = flow.env.get_world_size()
        if self.world_size / args.num_devices_per_node > 1:
            self.config.enable_cudnn_conv_heuristic_search_algo(True)

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
        if self.return_pred_and_label:
            pred = logits.softmax()
        else:
            pred = None
            label = None
        loss.backward()
        return loss, pred, label


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
        image = image.to("cuda")
        label = label.to("cuda")
        logits = self.model(image)
        pred = logits.softmax()
        return pred, label
