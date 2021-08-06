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


def make_dirs(*pathes):
    for path in pathes:
        # dir path
        if not os.path.exists(path):
            os.makedirs(path)


def load_mnist(data_dir, transpose=True):
    if os.path.exists(data_dir):
        print("Found MNIST - skip download")
    else:
        print("not Found MNIST - start download")
        download_mnist(data_dir)

    fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-images-idx3-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float32)

    fd = open(os.path.join(data_dir, "t10k-labels-idx1-ubyte"))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float32)

    X = trX
    y = trY.astype(int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    if transpose:
        X = np.transpose(X, (0, 3, 1, 2))

    return (X - 127.5) / 127.5, y_vec


def download_mnist(data_dir):
    import subprocess

    os.mkdir(data_dir)
    url_base = "http://yann.lecun.com/exdb/mnist/"
    file_names = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for file_name in file_names:
        url = (url_base + file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir, file_name)
        cmd = ["curl", url, "-o", out_path]
        print("Downloading ", file_name)
        subprocess.call(cmd)
        cmd = ["gzip", "-d", out_path]
        print("Decompressing ", file_name)
        subprocess.call(cmd)


def to_numpy(x, mean=True):
    if mean:
        x = flow.mean(x)

    return x.numpy()


def to_tensor(x, grad=True, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.Tensor(x, requires_grad=grad, dtype=dtype)


def save_to_gif(path):
    anim_file = os.path.join(path, "dcgan.gif")
    with imageio.get_writer(anim_file, mode="I") as writer:
        filenames = glob.glob(os.path.join(path, "*image*.png"))
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    print("Save images gif to {} done.".format(anim_file))


def save_images(x, size, path):
    x = x.astype(np.float)
    fig = plt.figure(figsize=(4, 4))
    for i in range(size):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i, 0, :, :] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")
    plt.savefig(path)
    print("Save image to {} done.".format(path))

class BCELoss(flow.nn.Module):
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
            self.add_optimizer("Adam_D", optimizer)
            self.of_cross_entropy = loss
        
        def build(self, images, label1, label0, z):
            print(images.shape)
            print(self.generator(z).shape)
            print("begin building")
            g_out = self.generator(z)

            cat = flow.cat((images, g_out), dim=0)

            result = self.discriminator(cat)
            d_logits = result[:images.shape[0]]
            g_logits = result[images.shape[0]:]

            print("finish discriminator")
            print(d_logits.shape)
            print(label1.shape)
            d_loss_real = self.of_cross_entropy(d_logits, label1)
            #d_loss_real.backward(retain_graph=True)
            
            # train D with all-fake batch
            print("not finished")
            d_loss_fake = self.of_cross_entropy(g_logits, label0)
            #d_loss_fake.backward()
            print("finished")
            d_loss = d_loss_fake + d_loss_real

            d_loss.backward()
            print("backward finished")
            # self.optimizerD.step()
            # self.optimizerD.zero_grad()

            # 这里加上to numpy会报错
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
        self.add_optimizer("Adam_D", optimizer)
        self.of_cross_entropy = loss

    def build(self, label1, z):
        g_out = self.generator(z)
        g_logits = self.discriminator(g_out)

        g_loss = self.of_cross_entropy(g_logits, label1)
        g_loss.backward()

        return (to_numpy(g_loss), to_numpy(g_out, False), to_numpy(g_logits))

class GeneratorEvalGraph(flow.nn.Graph):
    
    def __init__(self, g, z):
        super().__init__()
        self.generator = g
        self.fixed_z = z

    def build(self):
        with flow.no_grad():
            return self.generator(self.fixed_z)

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

        self.of_cross_entropy = BCELoss().to(self.device)
        #self.of_cross_entropy = flow.nn.CrossEntropyLoss().to(self.device)
        
        G_train_graph = GeneratorTrainGraph(self.discriminator, self.generator, self.optimizerG, self.of_cross_entropy)
        D_train_graph = DiscriminatorTrainGraph(self.discriminator, self.generator, self.optimizerD, self.of_cross_entropy)
        G_eval_graph = GeneratorEvalGraph(self.generator, self.generate_noise())

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
                    #d_loss = D_train_graph(images, label1_smooth, label0, self.generate_noise())
                    (
                        d_loss,
                        d_loss_fake,
                        d_loss_real,
                        D_x,
                        D_gz1,
                    ) = D_train_graph(images, label1_smooth, label0, self.generate_noise())
                else:
                    #d_loss = D_train_graph(images, label1_smooth, label0, self.generate_noise())
                    (
                        d_loss,
                        d_loss_fake,
                        d_loss_real,
                        D_x,
                        D_gz1,
                    ) = D_train_graph(images, label1, label0, self.generate_noise())
                g_loss, g_out, D_gz2 = G_train_graph(label1, self.generate_noise())

                if (batch_idx + 1) % 10 == 0:
                    self.G_loss.append(g_loss)
                    self.D_loss.append(d_loss)

                if (batch_idx + 1) % self.eval_interval == 0:
                    print(
                        "{}th epoch, {}th batch, d_fakeloss:{:>8.10f}, d_realloss:{:>8.10f}, d_loss:{:>8.10f}, g_loss:{:>8.10f}, D_x:{:>8.10f}, D_Gz:{:>8.10f} / {:>8.10f}".format(
                            epoch_idx + 1,
                            batch_idx + 1,
                            d_loss_fake[0],
                            d_loss_real[0],
                            d_loss[0],
                            g_loss[0],
                            D_x[0],
                            D_gz1[0],
                            D_gz2[0],
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
                os.path.join(self.path, "g_loss_{}.npy".format(epochs)), self.G_loss
            )
            np.save(
                os.path.join(self.path, "d_loss_{}.npy".format(epochs)), self.D_loss
            )

    def train_discriminator(self, images, label1, label0):
        # train D with all-real batch
        d_logits = self.discriminator(images)
        d_loss_real = self.of_cross_entropy(d_logits, label1)
        d_loss_real.backward(retain_graph=True)

        # train D with all-fake batch
        z = self.generate_noise()
        g_out = self.generator(z)
        g_logits = self.discriminator(g_out.detach())
        d_loss_fake = self.of_cross_entropy(g_logits, label0)
        d_loss_fake.backward()

        d_loss = d_loss_fake + d_loss_real
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

    dcgan = DCGAN(args)
    dcgan.train(args.epoch_num, args.save)


if __name__ == "__main__":
    args = _parse_args()
    main(args)
