import os
import time
import argparse
import numpy as np
import glob
import imageio
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import oneflow as flow

from utils import make_dirs, load_mnist, download_mnist, to_numpy, to_tensor, save_to_gif, save_images
from models import Generator, Discriminator, GeneratorTrainGraph, DiscriminatorTrainGraph, GeneratorEvalGraph

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
        self.G_loss_graph = []
        self.D_loss_graph = []

        self.path = args.path
        self.batch_size = args.batch_size
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        self.images_path = os.path.join(self.path, "images")
        self.train_images_path = os.path.join(self.images_path, "train_images")
        self.val_images_path = os.path.join(self.images_path, "val_images")
        make_dirs(self.checkpoint_path, self.train_images_path, self.val_images_path)

    
    
    def train(self, epochs=1, save=True):
        # init dataset
        np.random.seed(0)
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

        self.generator = Generator(self.z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        if args.load != "":
            self.generator.load_state_dict(flow.load(args.load))
            self.discriminator.load_state_dict(flow.load(args.load))

        self.optimizerG = flow.optim.SGD(self.generator.parameters(), lr=self.lr)
        self.optimizerD = flow.optim.SGD(self.discriminator.parameters(), lr=self.lr)

        self.of_cross_entropy = flow.nn.BCELoss().to(self.device)
        #self.of_cross_entropy = flow.nn.CrossEntropyLoss().to(self.device)
        
        self.generator_graph = Generator(self.z_dim).to(self.device)
        self.discriminator_graph = Discriminator().to(self.device)
        self.optimizerG_graph = flow.optim.SGD(self.generator_graph.parameters(), lr=self.lr)
        self.optimizerD_graph = flow.optim.SGD(self.discriminator_graph.parameters(), lr=self.lr)

        G_train_graph = GeneratorTrainGraph(self.discriminator_graph, self.generator_graph, self.optimizerG_graph, self.of_cross_entropy)
        D_train_graph = DiscriminatorTrainGraph(self.discriminator_graph, self.generator_graph, self.optimizerD_graph, self.of_cross_entropy)

        for epoch_idx in range(epochs):
            self.generator.train()
            self.discriminator.train()
            start = time.time()
            graph_time = 0
            normal_time = 0
            for batch_idx in range(batch_num):
                images = to_tensor(
                    x[
                        batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                    ].astype(np.float32)
                ).to(self.device)
                # one-side label smooth
                start = time.time()
                (
                    d_loss,
                    d_loss_fake,
                    d_loss_real,
                    D_x,
                    D_gz1,
                ) = D_train_graph(images, label1_smooth, label0, self.generate_noise())
                (
                    d_loss,
                    d_loss_fake,
                    d_loss_real,
                    D_x,
                    D_gz1,
                ) = (to_numpy(d_loss), to_numpy(d_loss_fake), to_numpy(d_loss_real), to_numpy(D_x), to_numpy(D_gz1))
                g_loss, g_out, D_gz2 = G_train_graph(label1, self.generate_noise())
                g_loss, g_out, D_gz2 = to_numpy(g_loss), to_numpy(g_out, False), to_numpy(D_gz2)
                graph_time += time.time() - start
                self.D_loss_graph.append(d_loss)
                self.G_loss_graph.append(g_loss)
                start = time.time()
                (
                    d_loss,
                    d_loss_fake,
                    d_loss_real,
                    D_x,
                    D_gz1,
                ) = self.train_discriminator(images, label1_smooth, label0)
                g_loss, g_out, D_gz2 = self.train_generator(label1)
                self.D_loss.append(d_loss)
                self.G_loss.append(g_loss)
                normal_time += time.time() - start

                if (batch_idx + 1) % self.eval_interval == 0:
                    print("graph time: ", graph_time / self.eval_interval)
                    print("normal time: ", normal_time / self.eval_interval)
                    graph_time = 0
                    normal_time = 0
                    np.save(
                        os.path.join(self.path, "g_loss.npy".format(epochs)), self.G_loss
                    )
                    np.save(
                        os.path.join(self.path, "d_loss.npy".format(epochs)), self.D_loss
                    )
                    np.save(
                        os.path.join(self.path, "g_loss_graph.npy".format(epochs)), self.G_loss_graph
                    )
                    np.save(
                        os.path.join(self.path, "d_loss_graph.npy".format(epochs)), self.D_loss_graph
                    )
                    return


    def train_discriminator(self, images, label1, label0):
        z = self.generate_noise()
        z = flow.zeros_like(z)
        g_out = self.generator(z)

        cat = flow.cat((images, g_out), dim=0)

        result = self.discriminator(cat)
        d_logits = result[:images.shape[0]]
        g_logits = result[images.shape[0]:]

        d_loss_real = self.of_cross_entropy(d_logits, label1)

        d_loss_fake = self.of_cross_entropy(g_logits, label0)

        d_loss = d_loss_fake + d_loss_real

        d_loss.backward()
        self.optimizerD.step()
        self.optimizerD.zero_grad()

        return (
            to_numpy(d_loss),
            to_numpy(d_loss_fake),
            to_numpy(d_loss_real),
            to_numpy(d_logits),
            to_numpy(g_logits),
        )

    def train_generator(self, label1):
        z = self.generate_noise()
        z = flow.zeros_like(z)
        g_out = self.generator(z)
        g_logits = self.discriminator(g_out)
        g_loss = self.of_cross_entropy(g_logits, label1)
        g_loss.backward()
        self.optimizerG.step()
        self.optimizerG.zero_grad()

        return (to_numpy(g_loss), to_numpy(g_out, False), to_numpy(g_logits))

    def generate_noise(self):
        return to_tensor(
            np.random.normal(0, 1, size=(self.batch_size, self.z_dim)), False
        ).to(self.device)

    def _eval_generator_and_save_images(self, epoch_idx):
        results = to_numpy(self._eval_generator(), False)
        save_images(
            results,
            self.eval_size,
            os.path.join(self.val_images_path, "image_{:02d}.png".format(epoch_idx)),
        )

    def _eval_generator(self):
        self.generator.eval()
        with flow.no_grad():
            g_out = self.generator(self.fixed_z)
        return g_out


def main(args):
    np.random.seed(0)
    dcgan = DCGAN(args)
    dcgan.train(args.epoch_num, args.save)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
