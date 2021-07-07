import argparse
import time
import numpy as np

from tqdm import tqdm
import oneflow.experimental as flow
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from train_of_srgan import save_obj
from models.of_model import Generator
from utils.of_data_utils import ValDatasetFromFolder
flow.enable_eager_execution()


def to_tensor(x, grad=True, dtype=flow.float32):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return flow.Tensor(x, requires_grad=grad, dtype=dtype)


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int,
                    help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str,
                    choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--data_dir', default='./data0/VOC2012',
                    type=str, help='data root')
parser.add_argument('--image_path', default='data/Set14/LR_bicubic/X4/monarchx4.png', type=str,
                    help='test low resolution image path')
parser.add_argument('--hr_path', default='data/Set14/HR/monarch.png', type=str,
                    help='test low resolution image path')
parser.add_argument('--save_image', default='data/Set14/SR/X4/monarchx4-oneflow.png', type=str,
                    help='super resolution image path')
parser.add_argument('--model_path', default='SRGAN_netG_epoch_4_99',
                    type=str, help='generator model epoch path')
opt = parser.parse_args()

TEST_MODE = True if opt.test_mode == 'GPU' else False
UPSCALE_FACTOR = opt.upscale_factor
MODEL_PATH = opt.model_path

netG = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    netG.to('cuda')
netG.load_state_dict(flow.load(MODEL_PATH))

modes = ["val", "train"]
val_data = ValDatasetFromFolder(
    opt.data_dir, mode=modes[1], upscale_factor=4)

start = time.process_time()
with flow.no_grad():
    val_bar = tqdm(val_data)
    valing_results = {'psnrs': 0, 'ssims': 0,
                      'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    for val_hr, val_lr in val_bar:
        batch_size = val_lr.shape[0]
        valing_results['batch_sizes'] += batch_size
        lr = to_tensor(val_lr)
        hr = to_tensor(val_hr)
        lr = lr.to('cuda')
        hr = hr.to('cuda')
        sr = netG(lr)
        fake = sr[0].data * 255
        fake = fake.squeeze(0).permute(1, 2, 0)
        fake = fake.numpy().astype(np.uint8)

        _img = hr[0].data * 255
        _img = _img.squeeze(0).permute(1, 2, 0)
        _img = _img.numpy().astype(np.uint8)

        batch_psnr = peak_signal_noise_ratio(_img, fake)
        valing_results['psnrs'] += batch_psnr * batch_size
        valing_results['psnr'] = valing_results['psnrs'] / \
            valing_results['batch_sizes']

        batch_ssim = structural_similarity(_img, fake, multichannel=True)
        valing_results['ssims'] += batch_ssim * batch_size
        valing_results['ssim'] = valing_results['ssims'] / \
            valing_results['batch_sizes']
        val_bar.set_description(
            desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                valing_results['psnr'], valing_results['ssim']))

save_obj(valing_results, "val_results")
