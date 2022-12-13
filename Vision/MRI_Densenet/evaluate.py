import os
import time
import numpy as np
import oneflow as flow
from utils import patching,depatching,ssim, psnr, nrmse
from matplotlib import pyplot as plt
import nibabel as nib
from model import Generator

# Set pathes
vm_PATH = "/data2/lijiayang/MRI/HCP/"
id_csv = './csv/id_hcp.csv'

# Number of workers for dataloader
workers = 1
# Batch size. It controls the number of samples once download
batch_size = 16
# Patch size, it controls the number of patches once send into the model
patch_size = 2
# The size of one image patch (eg. 64 means a cubic patch with size: 64x64x64)
cube_size = 64
# Set the usage of a patch cluster.
usage = 1.0
# Number of mDCSRN (G) pre-training steps (5e6)
num_steps_pre = 250000
# Number of WGAN training steps (1.5e7)
max_step=500000
# Learning rate for mDCSRN (G) pre-training optimizers
lr_pre = 1e-4
# Learning rate for optimizers
lr = 5e-6
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 4

model_path = '/home/lijiayang/Oneflow/epoches'

data_path = '/data2/lijiayang/MRI/HCP/100206_3T_Structural_unproc_3T_T1w_MPR1.nii'

# set GPU device
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device = flow.device("cuda:1" if (flow.cuda.is_available() and ngpu > 0) else "cpu")

netG = Generator(ngpu).cuda(device)
since = time.time()
netG.load_state_dict(flow.load(model_path))
netG.eval()
nii = nib.load(data_path)
image = np.array(nii.dataobj)
image = flow.ShortTensor(image)
image = image.unsqueeze(0)
print(image.shape)
patch_loader=patching(image, image,
               patch_size = patch_size,
               cube_size = cube_size,
               usage=1.0, is_training=False)
sr_data_cat = flow.Tensor([])
i = 1
for lr_patches, hr_patches in patch_loader:
    hr_patches=hr_patches.cuda(device)
    with flow.no_grad():
        sr_patches = netG(hr_patches)
        sr_data_cat = flow.cat([sr_data_cat, sr_patches.to("cpu")],0)
        print(i)
        i+=1

# calculate the evaluation metric
sr_data = depatching(sr_data_cat, image.size(0)).squeeze(0).cpu().numpy()
image = image.squeeze(0).cpu().numpy()
print(sr_data.shape)
for i in range(10):
    plt.imshow(image[128+i,:,:],cmap='gray')
    plt.imshow(sr_data[128+i,:,:],cmap='gray')
    plt.show()
