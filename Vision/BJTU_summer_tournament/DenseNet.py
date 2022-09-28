# 模型文件，使用预训练模型
import oneflow.nn as nn


def DenseNet121_pre(num_classes=10, pretrained=True):
    from flowvision.models import densenet121
    model = densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, num_classes)
    return model


def DenseNet161_pre(num_classes=10, pretrained=True):
    from flowvision.models import densenet161
    model = densenet161(pretrained=pretrained)
    model.classifier = nn.Linear(2208, num_classes)
    return model


def DenseNet169_pre(num_classes=10, pretrained=True):
    from flowvision.models import densenet169
    model = densenet169(pretrained=pretrained)
    model.classifier = nn.Linear(1664, num_classes)
    return model


def DenseNet201_pre(num_classes=10, pretrained=True):
    from flowvision.models import densenet201
    model = densenet201(pretrained=pretrained)
    model.classifier = nn.Linear(1920, num_classes)
    return model


if __name__ == '__main__':
    model = DenseNet121_pre(pretrained=False)
    print(model)
