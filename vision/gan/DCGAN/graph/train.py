import os
import time
import argparse
import numpy as np
import matplotlib

matplotlib.use("agg")
import oneflow as flow

from utils import (
    make_dirs,
    load_mnist,
    download_mnist,
    to_numpy,
    to_tensor,
    save_to_gif,
    save_images,
)
from models import (
    Generator,
    Discriminator,
    GeneratorTrainGraph,
    DiscriminatorTrainGraph,
    GeneratorEvalGraph,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="oneflow DCGAN")
    parser.add_argument("--path", type=str, default="./dcgan", required=False)
    parser.add_argument("-e", "--epoch_num", type=int, default=100, required=False)
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-4, required=False
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        required=False,
        help="the path to continue training the model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/mnist",
        required=False,
        help="the path to dataset",
    )
    parser.add_argument("--batch_size", type=int, default=256, required=False)
    parser.add_argument("--label_smooth", type=float, default=0.15, required=False)
    parser.add_argument(
        "--save",
        type=bool,
        default=True,
        required=False,
        help="whether to save train_images, train_checkpoint and train_loss",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="disables CUDA training"
    )
    return parser.parse_args()


class DCGAN(flow.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lr = args.learning_rate
        self.z_dim = 100
        self.eval_interval = 100
        self.eval_size = 16
        self.data_dir = args.data_dir
        self.device = "cpu" if args.no_cuda else "cuda"
        # evaluate generator based pn fixed noise during training
        self.fixed_z = to_tensor(
            np.random.normal(0, 1, size=(self.eval_size, self.z_dim)), False
        ).to(self.device)

        self.label_smooth = args.label_smooth
        self.G_loss = []
        self.D_loss = []

        self.path = args.path
        self.batch_size = args.batch_size
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.images_path = os.path.join(self.path, "images")
        self.train_images_path = os.path.join(self.images_path, "train_images")
        self.val_images_path = os.path.join(self.images_path, "val_images")
        make_dirs(self.checkpoint_path, self.train_images_path, self.val_images_path)

    def train(self, epochs=1, save=True):
        # init dataset
        x, _ = load_mnist(self.data_dir)
        batch_num = len(x) // self.batch_size
        label1 = to_tensor(np.ones(self.batch_size), False, dtype=flow.float32).to(
            self.device
        )
        label0 = flow.Tensor((np.zeros(self.batch_size)), dtype=flow.float32).to(
            self.device
        )
        if self.label_smooth != 0:
            label1_smooth = (label1 - self.label_smooth).to(self.device)

        # init training include optimizer, model, loss
        self.generator = Generator(self.z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        if args.load != "":
            self.generator.load_state_dict(flow.load(args.load))
            self.discriminator.load_state_dict(flow.load(args.load))

        self.optimizerG = flow.optim.Adam(self.generator.parameters(), lr=self.lr)
        self.optimizerD = flow.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.of_cross_entropy = flow.nn.BCELoss().to(self.device)

        G_train_graph = GeneratorTrainGraph(
            self.discriminator, self.generator, self.optimizerG, self.of_cross_entropy
        )
        D_train_graph = DiscriminatorTrainGraph(
            self.discriminator, self.generator, self.optimizerD, self.of_cross_entropy
        )
        self.G_eval_graph = GeneratorEvalGraph(self.generator)
        for epoch_idx in range(epochs):
            self.generator.train()
            self.discriminator.train()
            start = time.time()
            for batch_idx in range(batch_num):
                images = to_tensor(
                    x[
                        batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                    ].astype(np.float32)
                ).to(self.device)
                # one-side label smooth
                if self.label_smooth != 0:
                    (d_loss, d_loss_fake, d_loss_real, D_x, D_gz1,) = D_train_graph(
                        images, label1_smooth, label0, self.generate_noise()
                    )
                else:
                    (d_loss, d_loss_fake, d_loss_real, D_x, D_gz1,) = D_train_graph(
                        images, label1, label0, self.generate_noise()
                    )
                (d_loss, d_loss_fake, d_loss_real, D_x, D_gz1,) = (
                    to_numpy(d_loss),
                    to_numpy(d_loss_fake),
                    to_numpy(d_loss_real),
                    to_numpy(D_x),
                    to_numpy(D_gz1),
                )
                g_loss, g_out, D_gz2 = G_train_graph(label1, self.generate_noise())
                g_loss, g_out, D_gz2 = (
                    to_numpy(g_loss),
                    to_numpy(g_out, False),
                    to_numpy(D_gz2),
                )

                if (batch_idx + 1) % 100 == 0:
                    self.G_loss.append(g_loss)
                    self.D_loss.append(d_loss)

                if (batch_idx + 1) % self.eval_interval == 0:
                    print(
                        "{}th epoch, {}th batch, d_fakeloss:{:>8.10f}, d_realloss:{:>8.10f}, d_loss:{:>8.10f}, g_loss:{:>8.10f}, D_x:{:>8.10f}, D_Gz:{:>8.10f} / {:>8.10f}".format(
                            epoch_idx + 1,
                            batch_idx + 1,
                            d_loss_fake,
                            d_loss_real,
                            d_loss,
                            g_loss,
                            D_x,
                            D_gz1,
                            D_gz2,
                        )
                    )

            # save images based on .train()
            save_images(
                g_out,
                self.eval_size,
                os.path.join(
                    self.train_images_path, "fakeimage_{:02d}.png".format(epoch_idx)
                ),
            )

            # save images based on .eval()
            self._eval_generator_and_save_images(epoch_idx + 1)

            print(
                "Time for epoch {} is {} sec.".format(
                    epoch_idx + 1, time.time() - start
                )
            )

        if save:
            flow.save(
                self.generator.state_dict(),
                os.path.join(self.checkpoint_path, "g_{}".format(epoch_idx)),
            )
            flow.save(
                self.discriminator.state_dict(),
                os.path.join(self.checkpoint_path, "d_{}".format(epoch_idx)),
            )

            save_to_gif(self.train_images_path)
            save_to_gif(self.val_images_path)
            np.save(
                os.path.join(self.path, "g_loss_{}_graph.npy".format(epochs)),
                self.G_loss,
            )
            np.save(
                os.path.join(self.path, "d_loss_{}_graph.npy".format(epochs)),
                self.D_loss,
            )

    def generate_noise(self):
        return to_tensor(
            np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), False
        ).to(self.device)

    def _eval_generator_and_save_images(self, epoch_idx):
        results = to_numpy(self.G_eval_graph(self.fixed_z), False)
        save_images(
            results,
            self.eval_size,
            os.path.join(self.val_images_path, "image_{:02d}.png".format(epoch_idx)),
        )


def main(args):
    np.random.seed(0)
    dcgan = DCGAN(args)
    dcgan.train(args.epoch_num, args.save)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
