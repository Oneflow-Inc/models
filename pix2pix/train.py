import os
import numpy as np
import time
import argparse
import oneflow.experimental as flow
from models.networks import Generator, Discriminator, get_scheduler
from utils.dataset import load_facades
# from models.pix2pix import Pix2PixModel
from utils.utils import init_logger, to_tensor, to_numpy, save_images, mkdirs

os.environ["CUDA_VISIBLE_DEVICES"]= '2'

class Pix2Pix:
    def __init__(self, args) -> None:
        self.lr = args.learning_rate
        self.LAMBDA = args.LAMBDA
        self.save = args.save
        self.batch_size = args.batch_size
        self.path = args.path
        self.n_epochs = args.epoch_num
        self.eval_interval = 10
        self.G_image_loss = []
        self.G_GAN_loss = []
        self.G_total_loss = []
        self.D_loss = []
        self.netG = Generator().to("cuda")
        self.netD = Discriminator().to("cuda")
        self.optimizerG = flow.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.optimizerD = flow.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.criterionGAN = flow.nn.BCEWithLogitsLoss()
        # self.criterionGAN = flow.nn.BCELoss(reduction="mean")
        self.criterionL1 = flow.nn.L1Loss()
        self.schedulerG = get_scheduler(self.optimizerG, self.n_epochs)
        self.schedulerD = get_scheduler(self.optimizerD, self.n_epochs)

        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        # self.train_images_path = os.path.join(self.path, "train_images")
        self.test_images_path = os.path.join(self.path, "test_images")

        mkdirs(self.checkpoint_path, self.test_images_path, self.train_images_path)
        self.logger = init_logger(os.path.join(self.path, 'log.txt'))

    
    def train(self):
        # init dataset
        x, y = load_facades()
        # flow.Tensor() bug in here
        x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        self.fixed_inp = to_tensor(x[:self.batch_size].astype(np.float32))
        self.fixed_target = to_tensor(y[:self.batch_size].astype(np.float32))
        
        batch_num = len(x) // self.batch_size
        label1 = to_tensor(np.ones((self.batch_size, 1, 30, 30)), dtype=flow.float32)
        label0 = to_tensor(np.zeros((self.batch_size, 1, 30, 30)), dtype=flow.float32)
        
        for epoch_idx in range(self.n_epochs):
            self.netG.train()
            self.netD.train()
            start = time.time()
            
            # run every epoch to shuffle
            for batch_idx in range(batch_num):
                inp = to_tensor(x[
                    batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
                ].astype(np.float32))
                target = to_tensor( y[
                    batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
                ].astype(np.float32))

                # set_requires_grad(self.netD,True)
                # set_requires_grad(self.netG,False)
                # update D
                d_fake_loss, d_real_loss, d_loss = self.train_discriminator(inp, target, label0, label1)
                
                # set_requires_grad(self.netD,False)
                # set_requires_grad(self.netG,True)
                # update G
                g_gan_loss, g_image_loss, g_total_loss, g_out = self.train_generator(inp, target, label1)

                self.G_GAN_loss.append(g_gan_loss)
                self.G_image_loss.append(g_image_loss)
                self.G_total_loss.append(g_total_loss)
                self.D_loss.append(d_loss)
                if (batch_idx + 1) % self.eval_interval == 0:
                    self.logger.info(
                        "{}th epoch, {}th batch, d_fakeloss:{:>8.4f}, d_realloss:{:>8.4f},  ggan_loss:{:>8.4f}, gl1_loss:{:>8.4f}".format(
                            epoch_idx + 1, batch_idx + 1, d_fake_loss[0], d_real_loss[0], g_gan_loss[0], g_image_loss[0]
                        )
                    )

            self.logger.info("Time for epoch {} is {} sec.".format(
                epoch_idx + 1, time.time() - start))

            if (epoch_idx + 1) % 2 *self.eval_interval == 0:
                # save .train() images
                # save_images(g_out, to_numpy(inp, False), to_numpy(target, False), os.path.join(self.train_images_path, "trainimage_{:02d}.png".format(epoch_idx + 1)))
                # save .eval() images
                self._eval_generator_and_save_images(epoch_idx)

        if self.save:
            flow.save(self.netG.state_dict(),
                os.path.join(self.checkpoint_path, "pix2pix_g_{}".format(
                    epoch_idx + 1)
                )
            )

            flow.save(self.netD.state_dict(),
                os.path.join(self.checkpoint_path, "pix2pix_d_{}".format(
                    epoch_idx + 1)
                )
            )

            # save train loss and val error to plot
            np.save(os.path.join(
                self.path, 'G_image_loss_{}.npy'.format(self.n_epochs)), self.G_image_loss)
            np.save(os.path.join(
                self.path, 'G_GAN_loss_{}.npy'.format(self.n_epochs)), self.G_GAN_loss)
            np.save(os.path.join(
                self.path, 'G_total_loss_{}.npy'.format(self.n_epochs)), self.G_total_loss)
            np.save(os.path.join(self.path, 'D_loss_{}.npy'.format(self.n_epochs)), self.D_loss)
            self.logger.info("*************** Train done ***************** ")

    def train_generator(self, input, target, label1):
        g_out = self.netG(input)
        # First, G(A) should fake the discriminator
        fake_AB = flow.cat([input, g_out], 1)
        pred_fake = self.netD(fake_AB)
        gan_loss = self.criterionGAN(
            pred_fake, label1
        )
        # Second, G(A) = B
        l1_loss = self.criterionL1(g_out, target)
        # combine loss and calculate gradients
        g_loss = gan_loss + self.LAMBDA * l1_loss
        g_loss.backward()

        self.optimizerG.step()
        self.optimizerG.zero_grad()
        return to_numpy(gan_loss), to_numpy(self.LAMBDA * l1_loss), to_numpy(g_loss), to_numpy(g_out, False)

    def train_discriminator(self, input, target, label0, label1):
        g_out = self.netG(input)
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = flow.cat([input, g_out.detach()], 1)
        pred_fake = self.netD(fake_AB)
        d_fake_loss = self.criterionGAN(
            pred_fake, label0
        )

        # Real
        real_AB = flow.cat([input, target], 1)
        pred_real = self.netD(real_AB)
        d_real_loss = self.criterionGAN(
            pred_real, label1
        )

        # combine loss and calculate gradients
        d_loss = (d_fake_loss + d_real_loss)*0.5
        d_loss.backward()
        self.optimizerD.step()
        self.optimizerD.zero_grad()
        return to_numpy(d_fake_loss), to_numpy(d_real_loss), to_numpy(d_loss)

    def _eval_generator_and_save_images(self, epoch_idx):
        results = self._eval_generator()
        save_images(results, to_numpy(self.fixed_inp, False), to_numpy(self.fixed_target, False), path=os.path.join(self.test_images_path, "testimage_{:02d}.png".format(epoch_idx + 1))
        )

    def _eval_generator(self):
        self.netG.eval()
        with flow.no_grad():
            g_out = self.netG(self.fixed_inp)
        return to_numpy(g_out, False)


if __name__ == "__main__":
    flow.enable_eager_execution()
    parser = argparse.ArgumentParser(description="oneflow PIX2PIX")
    parser.add_argument("--path", type=str, default='./of_pix2pix', required=False)
    parser.add_argument("-e", "--epoch_num", type=int,
                        default=200, required=False)
    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=2e-4, required=False)
    parser.add_argument("--LAMBDA", type=float, default=200, required=False)
    parser.add_argument("--load", type=str, default="", required=False,
                        help="the path to continue training the model")
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--save", type=bool,
                        default=True, required=False, help="whether to save train_images, train_checkpoint and train_loss")
    args = parser.parse_args()

    pix2pix = Pix2Pix(args)
    pix2pix.train()
