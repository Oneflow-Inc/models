import oneflow as flow
import oneflow.nn.functional as F
from flowvision.utils import vision_helpers as vutils
import numpy as np
import os
import argparse
from tqdm import tqdm
from models import Generator


def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with flow.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return flow.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--artifacts', type=str, default=".", help='path to artifacts.')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=6)
    parser.add_argument('--end_iter', type=int, default=10)

    parser.add_argument('--dist', type=str, default='.')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=1024)
    parser.set_defaults(big=False)
    args = parser.parse_args()

    noise_dim = 256
    device = flow.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)

    for epoch in [10000*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = f"{args.artifacts}/model/{epoch}.pth"
        checkpoint = flow.load(ckpt)
        # Remove prefix `module`.
        # checkpoint['g'] = {k.replace('module.', ''): v for k, v in checkpoint['g'].items()}
        net_ig.load_state_dict(checkpoint['g'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, epoch %d'%epoch)

        net_ig.to(device)

        del checkpoint

        dist = 'eval_%d'%(epoch)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        with flow.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = flow.tensor(np.random.normal(size=(args.batch, noise_dim)), dtype=flow.float32).cuda()
                g_imgs = net_ig(noise)[0]
                g_imgs = F.interpolate(g_imgs, 512)
                for j, g_img in enumerate(g_imgs):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
