import oneflow as flow
import oneflow.nn as nn
import oneflow.nn.functional as F


class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features)),
        self.add_module('elu', nn.ELU(inplace=True)),
        self.add_module('conv', nn.Conv3d(num_input_features, growth_rate,
                                          kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        # Concatenation
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return flow.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate=growth_rate,
                                bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class Generator(nn.Module):

    def __init__(self, ngpu, growth_rate=16, block_config=(4, 4, 4, 4),
                 bn_size=2, drop_rate=0):

        super(Generator, self).__init__()
        # First convolution
        self.conv0 = nn.Conv3d(1, 2 * growth_rate, kernel_size=3, padding=1, bias=False)

        # Each denseblock
        num_features = 2 * growth_rate  # 2k
        num_features_cat = num_features
        self.block0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[0] * growth_rate + num_features
        self.comp0 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)

        self.block1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[1] * growth_rate + num_features
        self.comp1 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)

        self.block2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[2] * growth_rate + num_features
        self.comp2 = nn.Conv3d(num_features_cat, num_features,
                               kernel_size=1, stride=1, bias=False)

        self.block3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features,
                                  bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        num_features_cat += block_config[3] * growth_rate + num_features
        self.recon = nn.Conv3d(num_features_cat, 1,
                               kernel_size=1, stride=1, bias=False)

        # Official init from flow repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        out = self.block0(x)
        features = flow.cat([x, out], 1)
        out = self.comp0(features)

        out = self.block1(out)
        features = flow.cat([features, out], 1)
        out = self.comp1(features)

        out = self.block2(out)
        features = flow.cat([features, out], 1)
        out = self.comp2(features)

        out = self.block3(out)
        features = flow.cat([features, out], 1)
        out = self.recon(features)
        return out
