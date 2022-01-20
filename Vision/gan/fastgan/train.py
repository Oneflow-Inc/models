import argparse
import os
from sys import path
from flowvision.datasets.folder import ImageFolder
from numpy import dtype
import oneflow as flow
from config import get_config
from flowvision import transforms
from flowvision.utils import vision_helpers as vutils
from oneflow.utils.data.dataloader import DataLoader
from operation import ImageFolder, InfiniteSamplerWrapper
from models import Discriminator, Generator
import oneflow.optim as optim
import oneflow.nn as nn
from tqdm import tqdm
import numpy as np
from diffaug import DiffAugment
policy = 'color,translation'
import random
import oneflow.nn.functional as F
import lpips

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data = par1[k].data.mul(decay).add((1 - decay)*par2[k].data)


def train(netG, netD, netG_ema, optimizerG, optimizerD, args, cfg, dataloader, current_iteration, percept):
    fixed_noise = flow.tensor(np.random.normal(size=(8, cfg.G.nz)), dtype=flow.float32).cuda()

    for iteration in tqdm(range(current_iteration, cfg.TRAIN.iter)):
        real_image = next(dataloader)
        real_image = real_image.cuda()
        real_image = DiffAugment(real_image, policy=policy)
        current_batch_size = real_image.size(0)
        noise = flow.tensor(np.random.normal(size=(current_batch_size, cfg.G.nz)), dtype=flow.float32)
        fake_images1, fake_images2 = netG(noise.cuda())
        fake_images = [fake_images1, fake_images2]
        ## 2. train Discriminator
        optimizerD.zero_grad()
        part = random.randint(0, 3)
        pred, rec_all, rec_small, rec_part = netD(real_image, 'real', part=part)
        err = F.relu(flow.rand(*pred.shape).cuda() * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(real_image, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(real_image, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2]) ).sum()
        err.backward()
        err_dr = pred.mean().item()
        for fi in fake_images:
            pred = netD(fi.detach(), "fake")
            err = F.relu(flow.rand(*pred.shape).cuda() * 0.2 + 0.8 + pred).mean()
            err.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()

        err_g.backward()
        optimizerG.step()

        accumulate(netG_ema, netG)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (cfg.TRAIN.save_interval*10) == 0:
            with flow.no_grad():
                vutils.save_image(netG_ema(fixed_noise)[0].add(1).mul(0.5), cfg.TRAIN.saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( flow.cat([
                        F.interpolate(real_image, 128), 
                        rec_all, rec_small,
                        rec_part]).add(1).mul(0.5), cfg.TRAIN.saved_image_folder+'/rec_%d.jpg'%iteration )

        # if args.local_rank ==0:
        #     if iteration % (save_interval*50) == 0 or iteration == total_iterations:
        #         torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
        #         load_params(netG, backup_para)
        #         torch.save({'g':netG.state_dict(),
        #                     'd':netD.state_dict(),
        #                     'g_ema': avg_param_G,
        #                     'opt_g': optimizerG.state_dict(),
        #                     'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='region gan')
    parser.add_argument("--local_rank", type=int, default=0, required=False, help='local rank for DistributedDataParallel')
    args, unparsed = parser.parse_known_args()
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1
    cfg = get_config()
    cfg.TRAIN.nlr = cfg.TRAIN.nlr / n_gpu

    netG = Generator(ngf=cfg.G.ngf, nz=cfg.G.nz, im_size=cfg.TRAIN.im_size)
    # netG.apply(weights_init)
    netD = Discriminator(ndf=cfg.D.ndf, im_size=cfg.TRAIN.im_size)
    # netD.apply(weights_init)
    netG_ema = Generator(ngf=cfg.G.ngf, nz=cfg.G.nz, im_size=cfg.TRAIN.im_size)
    accumulate(netG_ema, netG, 0)

    optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.nlr, betas=(cfg.TRAIN.nbeta1, cfg.TRAIN.nbeta2))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.nlr, betas=(cfg.TRAIN.nbeta1, cfg.TRAIN.nbeta2))

    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True,)

    current_iteration = 0
    if cfg.TRAIN.checkpoint != 'None':
        pass

    if not os.path.isdir(cfg.TRAIN.saved_image_folder):
        os.mkdir(cfg.TRAIN.saved_image_folder)

    netG = netG.cuda()
    netD = netD.cuda()
    netG_ema = netG_ema.cuda()
    if args.distributed:
        netG = nn.parallel.DistributedDataParallel(
            netG,
            broadcast_buffers=False,
        )

        netD = nn.parallel.DistributedDataParallel(
            netD,
            broadcast_buffers=False,
        )


    transform_list = [
        transforms.Resize((int(cfg.TRAIN.im_size), int(cfg.TRAIN.im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in cfg.DATA.DATA_PATH:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(cfg.DATA.DATA_PATH, trans, 1024)
    else:
        dataset = ImageFolder(root=cfg.DATA.DATA_PATH, transform=trans)
    
    dataloader = iter(DataLoader(dataset, batch_size=cfg.DATA.BATCH_SIZE, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=cfg.TRAIN.dataloader_workers,))
    
    train(netG, netD, netG_ema, optimizerG, optimizerD, args, cfg, dataloader, current_iteration, percept)