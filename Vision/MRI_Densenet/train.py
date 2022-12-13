import os
import time
import math
import re
import oneflow as flow
import oneflow.nn as nn
import oneflow.optim as optim
from data_util import CustomDatasetFromCSV
from utils import patching,depatching,ssim, psnr, nrmse
from oneflow.utils.data import SubsetRandomSampler
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
pretrained=' '
# set GPU device
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
device = flow.device("cuda:1" if (flow.cuda.is_available() and ngpu > 0) else "cpu")
# Create the generator

train_split= 0.7
dataset = CustomDatasetFromCSV(id_csv,vm_PATH)
dataset_size = len(dataset)
indices = list(range(dataset_size))
train_size = math.ceil(train_split * dataset_size)
train_indices = indices[:train_size]
train_sampler = SubsetRandomSampler(train_indices)
train_loader = flow.utils.data.DataLoader(dataset=dataset,
                                        batch_size=batch_size,
                                        sampler=train_sampler,
                                        shuffle=False,
                                        num_workers=workers)

since = time.time()
netG = Generator(ngpu).cuda(device)
optimizer = optim.Adam(netG.parameters(), lr=lr)
criterion = nn.L1Loss()
print("Generator pre-training...")
if pretrained != ' ':
    netG.load_state_dict(flow.load(pretrained))
    # if transfer from a single gpu case, set multi-gpu here again.
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(netG, list(range(ngpu)))
    step = int(re.sub("\D", "", pretrained))  # start from the pretrained model's step
else:
    step = 0
while (step < max_step):
    print('Step {}/{}'.format(step, max_step))
    print('-' * 10)
    epoch_loss = 0.0
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            netG.train()  # Set model to training mode
        else:
            netG.eval()  # Set model to training mode
        batch_loss = 0.0

        for lr_data, hr_data in train_loader:
            patch_loader = patching(lr_data, hr_data,
                                    patch_size=patch_size,
                                    cube_size=cube_size,
                                    usage=usage, is_training=True)
            patch_count = 0
            patch_loss = 0.0
            for lr_patches, hr_patches in patch_loader:
                lr_patches = lr_patches.cuda(device)
                hr_patches = hr_patches.cuda(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                sr_patches = netG(lr_patches)
                loss = criterion(sr_patches, hr_patches)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    step += 1  # we count step here
                    # This print out is only for early inspection
                    #                             if (step % 500) == 0:
                    #                                 print('Step: {}, loss= {:.4f}'.format(step, loss.item()))
                    if (step % int(max_step // 10)) == 0:
                        # save intermediate models
                        flow.save(netG.state_dict(), 'epoches/trained_G_step{}'.format(step))

                    if (step == max_step):
                        print('Complete {} steps'.format(step))
                        flow.save(model.state_dict(), 'epoches/pretrained_G_step{}'.format(step))

                # statistics
                patch_count += lr_patches.size(0)
                patch_loss += loss.item() * lr_patches.size(0)
            batch_loss += patch_loss / patch_count
        epoch_loss = batch_loss / dataset_size[phase]
        print('Step: {}, {} Loss: {:.4f}'.format(step, phase, epoch_loss))

    time_elapsed = time.time() - since
    print('Now the training uses {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
