import numpy as np
import dataloader
from model.UNet import UNet
from visualize import visualize
import oneflow as flow
from oneflow._oneflow_internal import float32

net = UNet(n_channels=3, n_classes=1)

checkpoint_path = ".../result/UNetmodel-i"
checkpoint = flow.load(checkpoint_path)
net.load_state_dict(checkpoint)

device = "cpu"
net.to("cpu")

test_dataset = dataloader.Dataset(
    dataloader.x_test_dir,
    dataloader.y_test_dir,
    augmentation=dataloader.get_test_augmentation(),
)

for image, mask in test_dataset:
    show_image = image
    with flow.no_grad():
        image = image / 255.0
        image = image.astype("float32")
        image = flow.tensor(image, dtype=float32)
        image = image.permute(2, 0, 1)
        image = image.to(device)
        print(image.shape)

        pred = net(image.unsqueeze(0).to(device))
        pred = pred.numpy()
        pred = pred > 0.5

    visualize(image=show_image, GT=mask[0, :, :], Pred=pred[0, 0, :, :])
