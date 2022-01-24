import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from calc_inception import load_patched_inception_v3
import os

@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img,_ in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = torch.cat(feature_list, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--path_a', type=str)
    parser.add_argument('--path_b', type=str)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--end', type=int, default=13)

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            #transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = ImageFolder(args.path_a, transform)
    loader_a = DataLoader(dset_a, batch_size=args.batch, num_workers=4)

    features_a = extract_features(loader_a, inception, device).numpy()
    print(f'extracted {features_a.shape[0]} features')

    real_mean = np.mean(features_a, 0)
    real_cov = np.cov(features_a, rowvar=False)
    
    #for folder in os.listdir(args.path_b):
    for folder in range(args.iter,args.end+1):
        folder = 'eval_%d'%(folder*10000)
        if os.path.exists(os.path.join( args.path_b, folder )):
            print(folder)
            dset_b = ImageFolder( os.path.join( args.path_b, folder ), transform)
            loader_b = DataLoader(dset_b, batch_size=args.batch, num_workers=4)

            features_b = extract_features(loader_b, inception, device).numpy()
            print(f'extracted {features_b.shape[0]} features')

            sample_mean = np.mean(features_b, 0)
            sample_cov = np.cov(features_b, rowvar=False)

            fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

            print(folder, ' fid:', fid)
