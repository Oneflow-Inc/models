import os
import oneflow.experimental as flow
import numpy as np
import time
import argparse
from datetime import datetime
from models.networks import Generator, Discriminator, get_scheduler, set_requires_grad
from utils.dataset import load_facades
# from models.pix2pix import Pix2PixModel
from utils.utils import to_tensor, to_numpy

os.environ["CUDA_VISIBLE_DEVICES"]= '3'

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
        self.criterionL1 = flow.nn.L1Loss()
        self.schedulerG = get_scheduler(self.optimizerG, self.n_epochs)
        self.schedulerD = get_scheduler(self.optimizerD, self.n_epochs)

        if not os.path.exists(self.path):
            os.mkdir(self.path)
            print("Make new dir '{}' done.".format(self.path))
        self.checkpoint_path = os.path.join(self.path, "checkpoint")
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        # self.test_images_path = os.path.join(self.path, "test_images")
        # if not os.path.exists(self.test_images_path):
        #     os.mkdir(self.test_images_path)
    
    def train(self):
        # init dataset
        x, y = load_facades()
        
        batch_num = len(x) // self.batch_size
        label1 = to_tensor(np.ones((self.batch_size, 1, 30, 30)), dtype=flow.float32).to("cuda")
        label0 = to_tensor(np.zeros((self.batch_size, 1, 30, 30)), dtype=flow.float32).to("cuda")
        
        for epoch_idx in range(self.n_epochs):
            self.netG.train()
            self.netD.train()
            start = time.time()
            self.schedulerD.step()
            self.schedulerG.step()
            
            # run every epoch to shuffle
            for batch_idx in range(batch_num):
                inp = to_tensor(x[
                    batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
                ].astype(np.float32)).to("cuda")
                target = to_tensor(y[
                    batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
                ].astype(np.float32)).to("cuda")

                # G(A)
                g_out = self.netG(inp)

                set_requires_grad(self.netD,True)
                # update D
                d_loss = self.train_discriminator(g_out, inp, target, label0, label1)
                
                set_requires_grad(self.netD,False)
                set_requires_grad(self.netG,True)
                # update G
                g_gan_loss, g_image_loss, g_total_loss = self.train_generator(g_out,
                    inp, target, label1)

                self.G_GAN_loss.append(g_gan_loss)
                self.G_image_loss.append(g_image_loss)
                self.G_total_loss.append(g_total_loss)
                self.D_loss.append(d_loss)
                if (batch_idx + 1) % self.eval_interval == 0:
                    print(
                        "Train {}th epoch, {}th batch, dloss:{}, g_gan_loss:{}, g_image_loss:{}, g_total_loss:{}".format(
                            epoch_idx + 1, batch_idx + 1, d_loss[0], g_gan_loss[0], g_image_loss[0], g_total_loss[0]
                        )
                    )

            print("Time for epoch {} is {} sec.".format(
                epoch_idx + 1, time.time() - start))

        if self.save:
            flow.save(self.netG.state_dict(),
                os.path.join(self.checkpoint_path, "pix2pix_g_{}".format(
                    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
                )
            )

            flow.save(self.netD.state_dict(),
                os.path.join(self.checkpoint_path, "pix2pix_d_{}".format(
                    str(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
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
            print("*************** Train done ***************** ")

    def train_generator(self, g_out, input, target, label1):
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
        return to_numpy(gan_loss), to_numpy(l1_loss), to_numpy(g_loss)

    def train_discriminator(self, g_out, input, target, label0, label1):
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
        return to_numpy(d_loss)


if __name__ == "__main__":
    flow.enable_eager_execution()
    parser = argparse.ArgumentParser(description="oneflow PIX2PIX")
    parser.add_argument("--path", type=str, default='./', required=False)
    parser.add_argument("-e", "--epoch_num", type=int,
                        default=200, required=False)
    parser.add_argument("-lr", "--learning_rate",
                        type=float, default=2e-4, required=False)
    parser.add_argument("--LAMBDA", type=float, default=100, required=False)
    parser.add_argument("--load", type=str, default="", required=False,
                        help="the path to continue training the model")
    parser.add_argument("--batch_size", type=int, default=1, required=False)
    parser.add_argument("--save", type=bool,
                        default=True, required=False, help="whether to save train_images, train_checkpoint and train_loss")
    args = parser.parse_args()

    pix2pix = Pix2Pix(args)
    pix2pix.train()
