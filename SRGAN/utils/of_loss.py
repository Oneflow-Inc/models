import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import sys, os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)
from vgg.models.vgg import vgg19_bn, vgg16_bn, vgg19, vgg16

model_dict = {
    "vgg16": vgg16,
    "vgg19": vgg19,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
}


class GeneratorLoss(nn.Module):
    def __init__(self, path):
        super(GeneratorLoss, self).__init__()
        vgg = model_dict["vgg16"]()
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        pretrain_models = flow.load(path)
        loss_network.load_state_dict(
            {
                k.replace("features.", ""): v
                for k, v in pretrain_models.items()
                if "features" in k
            }
        )
        loss_network.to("cuda")
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = flow.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(
            self.loss_network(out_images), self.loss_network(target_images)
        )
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return (
            image_loss
            + 0.001 * adversarial_loss
            + 0.006 * perception_loss
            + 2e-8 * tv_loss
        )


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = flow.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = flow.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class BCELoss(nn.Module):
    def __init__(self, reduction: str = "mean", reduce=True) -> None:
        super().__init__()
        if reduce is not None and not reduce:
            raise ValueError("Argument reduce is not supported yet")
        assert reduction in [
            "none",
            "mean",
            "sum",
            None,
        ], "only 'sum', 'mean' and 'none' supported by now"

        self.reduction = reduction

    def forward(self, input, target, weight=None):
        assert (
            input.shape == target.shape
        ), "The Input shape must be the same as Target shape"

        _cross_entropy_loss = flow.negative(
            target * flow.log(input) + (1 - target) * flow.log(1 - input)
        )

        if weight is not None:
            assert (
                weight.shape == input.shape
            ), "The weight shape must be the same as Input shape"
            _weighted_loss = weight * _cross_entropy_loss
        else:
            _weighted_loss = _cross_entropy_loss

        if self.reduction == "mean":
            return flow.mean(_weighted_loss)
        elif self.reduction == "sum":
            return flow.sum(_weighted_loss)
        else:
            return _weighted_loss
