import numpy as np
import matplotlib.pyplot as plt
import oneflow as flow


class Generator(flow.nn.Module):
    def __init__(self, z_dim=100, dim=256) -> None:
        super().__init__()
        self.dim = dim
        self.input_fc = flow.nn.Sequential(
            flow.nn.Linear(z_dim, 7 * 7 * dim),
            flow.nn.BatchNorm1d(7 * 7 * dim),
            flow.nn.LeakyReLU(0.3),
        )
        self.model = flow.nn.Sequential(
            # (n, 128, 7, 7)
            flow.nn.ConvTranspose2d(dim, dim // 2, kernel_size=5, stride=1, padding=2),
            flow.nn.BatchNorm2d(dim // 2),
            flow.nn.LeakyReLU(0.3),
            # (n, 64, 14, 14)
            flow.nn.ConvTranspose2d(
                dim // 2, dim // 4, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            flow.nn.BatchNorm2d(dim // 4),
            flow.nn.LeakyReLU(0.3),
            # (n, 1, 28, 28)
            flow.nn.ConvTranspose2d(
                dim // 4, 1, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            flow.nn.Tanh(),
        )

    def forward(self, x):
        # (n, 256, 7, 7)
        x1 = self.input_fc(x).reshape((-1, self.dim, 7, 7))
        y = self.model(x1)

        return y


class Discriminator(flow.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = flow.nn.Sequential(
            flow.nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            flow.nn.LeakyReLU(0.3),
            flow.nn.Dropout(0.3),
            flow.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            flow.nn.LeakyReLU(0.3),
            flow.nn.Dropout(0.3),
        )

        self.fc = flow.nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        b = x.shape[0]
        x1 = self.model(x).reshape((b, -1))
        y = flow.sigmoid(self.fc(x1))
        return y.flatten()


class DiscriminatorTrainGraph(flow.nn.Graph):
    def __init__(self, d, g, optimizer, loss):
        super().__init__()
        self.discriminator = d
        self.generator = g
        self.add_optimizer(optimizer)
        self.of_cross_entropy = loss

    def build(self, images, label1, label0, z):
        g_out = self.generator(z)

        cat = flow.cat((images, g_out), dim=0)

        result = self.discriminator(cat)
        d_logits = result[: images.shape[0]]
        g_logits = result[images.shape[0] :]

        d_loss_real = self.of_cross_entropy(d_logits, label1)

        # train D with all-fake batch
        d_loss_fake = self.of_cross_entropy(g_logits, label0)

        d_loss = d_loss_fake + d_loss_real

        d_loss.backward()
        return (
            d_loss,
            d_loss_fake,
            d_loss_real,
            d_logits,
            g_logits,
        )


class GeneratorTrainGraph(flow.nn.Graph):
    def __init__(self, d, g, optimizer, loss):
        super().__init__()
        self.discriminator = d
        self.generator = g
        self.add_optimizer(optimizer)
        self.of_cross_entropy = loss

    def build(self, label1, z):
        g_out = self.generator(z)
        g_logits = self.discriminator(g_out)
        g_loss = self.of_cross_entropy(g_logits, label1)
        g_loss.backward()
        return (g_loss, g_out, g_logits)


class GeneratorEvalGraph(flow.nn.Graph):
    def __init__(self, g):
        super().__init__()
        self.generator = g

    def build(self, z):
        with flow.no_grad():
            pred = self.generator(z)
        return pred
