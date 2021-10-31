import oneflow as flow
from oneflow.nn import image
from eval.verification import test
from function import Train_Module
import time
import numpy as np

def make_static_grad_scaler():
    return flow.amp.StaticGradScaler(flow.env.get_world_size())


def make_grad_scaler():
    return flow.amp.GradScaler(
        init_scale=2 ** 30, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
    )

class SyntheticDataLoader(flow.nn.Module):
    def __init__(
        self, batch_size, image_size=112, num_classes=10000, placement=None, sbp=None,
    ):
        super().__init__()

        self.image_shape = (batch_size, 1024)
        self.label_shape = (batch_size,)
        self.num_classes = num_classes
        self.placement = placement
        self.sbp = sbp

        if self.placement is not None and self.sbp is not None:
            self.image = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=255,
                    size=self.image_shape,
                    dtype=flow.float32,
                    placement=self.placement,
                    sbp=self.sbp,
                ),
                requires_grad=False,
            )
            self.label = flow.nn.Parameter(
                flow.randint(
                    0,
                    high=self.num_classes,
                    size=self.label_shape,
                    placement=self.placement,
                    sbp=self.sbp,
                ).to(dtype=flow.int32),
                requires_grad=False,
            )
        else:
            self.image = flow.randint(
                0, high=255, size=self.image_shape, dtype=flow.float32, device="cuda"
            )
            self.label = flow.randint(
                0, high=self.num_classes, size=self.label_shape, device="cuda",
            ).to(dtype=flow.int32)

    def __len__(self):
        return 100

    def forward(self):
        return self.image, self.label


margin_softmax = flow.nn.CombinedMarginLoss(
    1, 0., 0.4).to("cuda")
of_cross_entropy = flow.nn.CrossEntropyLoss().to("cuda")


def make_optimizer(model):
    param_group = {"params": [p for p in model.parameters() if p is not None]}

    # if args.grad_clipping > 0.0:
    #     assert args.grad_clipping == 1.0, "ONLY support grad_clipping == 1.0"
    #     param_group["clip_grad_max_norm"] = (1.0,)
    #     param_group["clip_grad_norm_type"] = (2.0,)

    optimizer = flow.optim.SGD(
        [param_group],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
    )
    return optimizer


class TrainGraph(flow.nn.Graph):
    def __init__(
        self,
        model,
        fp16,
        data_loader,
        optimizer,
        lr_scheduler
    ):
        super().__init__()

        if fp16:
            self.config.enable_amp(True)
            self.set_grad_scaler(make_grad_scaler())
            self.set_grad_scaler(make_static_grad_scaler())

        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)

        self.model = model
        self.data_loader = data_loader
        self.cross_entropy = of_cross_entropy
        self.combine_margin = margin_softmax
        self.data_loader = data_loader
        self.add_optimizer(optimizer, lr_sch=lr_scheduler)

    def build(self, image, label):


        logits = self.model(image)
        logits = self.combine_margin(logits, label)*64
        loss= self.cross_entropy(logits, label) 
        loss.backward()
        return loss


class FC7(flow.nn.Module):
    def __init__(self, embedding_size, num_classes, bias=False):
        super(FC7, self).__init__()

        self.weight = flow.nn.Parameter(
            flow.empty(embedding_size, num_classes))
        flow.nn.init.normal_(self.weight, mean=0, std=0.01)

    def forward(self, x):
        x = flow.nn.functional.l2_normalize(input=x, dim=1, epsilon=1e-10)
        weight = self.weight
        weight = flow.nn.functional.l2_normalize(
            input=weight, dim=1, epsilon=1e-10)
        if x.is_consistent:
            x = x.to_consistent(sbp=flow.sbp.broadcast)
        x = flow.matmul(x, weight)
        return x




def test_model_parallel():
    placement = flow.env.all_device_placement("cpu")
    sbp = flow.sbp.split(0)
    batch_size = 36
    num_classes = 100000
    world_size = flow.env.get_world_size()
    embedding_size = 1024

    data_loader = SyntheticDataLoader(
        batch_size=batch_size,
        num_classes=num_classes,
        placement=placement,
        sbp=sbp,
    )

    input_size = embedding_size
    output_size = int(num_classes/world_size)
    train_module = FC7(input_size, output_size).to("cuda").to_consistent(
        placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.split(1))
      
    train_module.train()
    optimizer = make_optimizer(train_module)
    scheduler = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[5000], gamma=0.1
    )
    train_graph = TrainGraph(
        train_module, True, data_loader, optimizer, scheduler)
    image, labels = data_loader()
    image = image.to("cuda")
    labels = labels.to("cuda")
    loss_logits = train_graph(image, labels)
    time_start = time.time()
    for steps in range(len(data_loader)):
        loss = train_graph(image, labels)

    print("model_parallel: ",time.time()-time_start)


def test_model():

    placement = flow.env.all_device_placement("cpu")
    sbp = flow.sbp.split(0)
    batch_size = 36
    num_classes = 100000
    world_size = flow.env.get_world_size()
    embedding_size = 1024

    data_loader = SyntheticDataLoader(
        batch_size=batch_size,
        num_classes=num_classes,
        placement=placement,
        sbp=sbp,
    )
    train_module = FC7(embedding_size, num_classes=num_classes).to("cuda").to_consistent(
        placement=flow.env.all_device_placement("cuda"), sbp=flow.sbp.broadcast)

    optimizer = make_optimizer(train_module)
    scheduler = flow.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[5000], gamma=0.1
    )
    train_graph = TrainGraph(
        train_module, True, data_loader, optimizer, scheduler)
    image, labels = data_loader()

    image, labels = data_loader()
    image = image.to("cuda")
    labels = labels.to("cuda")
    loss = train_graph(image, labels)
    time_start = time.time()
    for steps in range(len(data_loader)):
        loss = train_graph(image, labels)

    print("model: ",time.time()-time_start)




if __name__ == "__main__":
    test_model_parallel()
    test_model()
