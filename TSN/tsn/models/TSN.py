import oneflow.experimental.nn as nn
from .resnet50 import resnet50
from .simple_consensus import SimpleConsensus
from .simple_spatial_module import SimpleSpatialModule
from .cls_head import ClsHead


class TSN(nn.Module):
    def __init__(
        self,
        spatial_feature_size,
        dropout_ratio,
        num_classes,
        pretrained=None,
        modality="RGB",
        in_channels=3,
        spatial_type="avg",
        spatial_size=7,
        consensus_type="avg",
    ):

        super(TSN, self).__init__()
        self.backbone = resnet50(pretrained=pretrained)
        self.modality = modality
        self.in_channels = in_channels
        self.spatial_temporal_module = SimpleSpatialModule(spatial_type, spatial_size)
        self.segmental_consensus = SimpleConsensus(consensus_type)
        self.cls_head = ClsHead(spatial_feature_size, dropout_ratio, num_classes)

        assert modality in ["RGB", "Flow", "RGBDiff"]

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        self.spatial_temporal_module.init_weights()
        self.segmental_consensus.init_weights()
        self.cls_head.init_weights()

    def forward(self, num_modalities, gt_label, img_group, return_loss=False):
        if return_loss:
            return self.forward_train(num_modalities, gt_label, img_group)
        else:
            return self.forward_test(num_modalities, gt_label, img_group)

    def forward_train(self, num_modalities, gt_label, img_group):
        assert num_modalities == 1

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + tuple(img_group.shape[3:])
        )
        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)

        x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + tuple(x.shape[1:]))

        x = self.segmental_consensus(x)
        x = x.squeeze(1)
        losses = dict()

        cls_score = self.cls_head(x)

        return cls_score

    def forward_test(self, num_modalities, gt_label, img_group):
        assert num_modalities == 1

        bs = img_group.shape[0]
        img_group = img_group.reshape(
            (-1, self.in_channels) + tuple(img_group.shape[3:])
        )

        num_seg = img_group.shape[0] // bs
        x = self.extract_feat(img_group)
        x = self.spatial_temporal_module(x)
        x = x.reshape((-1, num_seg) + tuple(x.shape[1:]))
        x = self.segmental_consensus(x)
        x = x.squeeze(1)
        x = self.cls_head(x)

        return x.numpy()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x
