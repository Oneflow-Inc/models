import torch.nn as nn
from .simple_spatial_module_torch import SimpleSpatialModule
from .simple_consensus_torch import SimpleConsensus
from .cls_head_torch import ClsHead
from .resnet_torch import resnet50


class TSN(nn.Module):
    def __init__(
        self,
        spatial_feature_size,
        dropout_ratio,
        num_classes,
        modality="RGB",
        in_channels=3,
        spatial_type="avg",
        spatial_size=7,
        consensus_type="avg",
    ):

        super(TSN, self).__init__()
        self.backbone = resnet50()
        self.modality = modality
        self.in_channels = in_channels
        #
        self.spatial_temporal_module = SimpleSpatialModule(spatial_type, spatial_size)
        self.segmental_consensus = SimpleConsensus(consensus_type)
        self.cls_head = ClsHead(spatial_feature_size, dropout_ratio, num_classes)

        assert modality in ["RGB", "Flow", "RGBDiff"]

        self.init_weights()

    def init_weights(self):
        pass
        self.backbone.init_weights()
        self.spatial_temporal_module.init_weights()
        self.segmental_consensus.init_weights()
        self.cls_head.init_weights()
