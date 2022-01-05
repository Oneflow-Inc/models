import imp
from sys import path
from flowvision.datasets.folder import ImageFolder
from numpy import dtype
import oneflow as flow
from config import get_config
from flowvision import transforms
from flowvision import utils as vutils
from oneflow.utils.data.dataloader import DataLoader
from operation import ImageFolder, InfiniteSamplerWrapper
from operation import copy_G_params, load_params, get_dir
from models import weights_init, Discriminator, Generator
import oneflow.optim as optim
import oneflow.nn as nn
from tqdm import tqdm
import numpy as np
from diffaug import DiffAugment
policy = 'color,translation'
import random
import oneflow.nn.functional as F

def train_d(net, data, label='real'):
    if label=='real':
        part = random.randint(0, 3)
        pred = [rec_all, rec_small, rec_part] = net(data, label, part=path)
        err = F.relu(flow.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu(flow.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()


if __name__ == '__main__':
    cfg = get_config()
    device = flow.device('cuda') if flow.cuda.is_available() else flow.device('cpu')
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
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=cfg.TRAIN.dataloader_workers, pin_memory=True))

    netG = Generator(ngf=cfg.G.ngf, nz=cfg.G.nz, im_size=cfg.TRAIN.im_size)
    netG.apply(weights_init)
    netD = Discriminator(ndf=cfg.D.ndf, im_size=cfg.TRAIN.im_size)
    netD.apply(weights_init)
    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = flow.tensor(np.random.normal(size=(8, cfg.G.nz)), dtype=flow.float32, device=device)

    optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.nlr, betas=(cfg.TRAIN.nbeta1, cfg.TRAIN.nbeta2))
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.nlr, betas=(cfg.TRAIN.nbeta1, cfg.TRAIN.nbeta2))

    if cfg.TRAIN.checkpoint != 'None':
        ckpt = flow.load(cfg.TRAIN.checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(cfg.TRAIN.checkpoint.split('_')[-1].split('.')[0])
        del ckpt
    
    if cfg.TRAIN.multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    for iteration in tqdm(range(current_iteration, cfg.TRAIN.iter)):
        real_image = next(dataloader)
        real_image = real_image.to(device)
        current_batch_size = real_image.size(0)
        noise = flow.tensor(np.random.normal(size=(current_batch_size, cfg.G.nz)), dtype=flow.float32, device=device)

        fake_images = netG(noise)
        real_image = DiffAugment(real_image, policy=policy)

        ## 2. train Discriminator
        netD.zero_grad()
        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()