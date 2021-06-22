import oneflow.experimental as flow
from oneflow.experimental import nn
from .networks import Generator, Discriminator, get_scheduler, set_requires_grad
from utils.utils import to_tensor, to_numpy

class Pix2PixModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.netG = Generator().to("cuda")
        self.netD = Discriminator().to("cuda")
        self.optimizerG = flow.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.optimizerD = flow.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5,0.999))
        self.criterionGAN = flow.nn.BCEWithLogitsLoss().to("cuda")
        self.criterionL1 = flow.nn.L1Loss().to("cuda")
        self.schedulerG = get_scheduler(self.optimizerG, self.n_epochs)
        self.schedulerD = get_scheduler(self.optimizerD, self.n_epochs)
    
    def set_input(self,input):
        pass

    def forward(self, input):
        g_out = self.netG(input)

    def set_requires_grad(self):
        pass

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = flow.cat([input, g_out], 1)
        pred_fake = self.netD(fake_AB.detach())

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
        d_loss = d_fake_loss + d_real_loss
        d_loss.backward()
        self.optimizerD.step()
        self.optimizerD.zero_grad()
        return to_numpy(d_loss)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
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

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights