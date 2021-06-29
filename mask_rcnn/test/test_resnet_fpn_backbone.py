import oneflow.experimental as flow
from models.backbone_utils import resnet_fpn_backbone
import numpy as np
flow.enable_eager_execution()
flow.InitEagerGlobalSession()


def test_resnet_fpn_backbone():
    backbone = resnet_fpn_backbone('resnet50', pretrained=True, trainable_layers=3)
    x = flow.Tensor(np.random.rand(1, 3, 64, 64))
    output = backbone(x)
    print([(k, v.shape) for k, v in output.items()])


if __name__ == '__main__':
    test_resnet_fpn_backbone()


