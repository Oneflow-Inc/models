import oneflow as flow
import numpy as np
import math
import flowvision
from flowvision import datasets, models, transforms
import time
import os
import subprocess
import pandas as pd
from oneflow.utils.data.dataset import Dataset
import scipy.io
import nibabel as nib


class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path, vm_path):
        self.data = pd.read_csv(csv_path)
        self.vm_path = vm_path
        self.is_transform = True

    def transform(self, image):
        '''
        This function transforms the 3D image of np.ndarray (z,x,y) to a flow.ShortTensor (B,z,x,y).

        '''
        image_flow = flow.ShortTensor(image)
        return image_flow

    def __getitem__(self, idx):
        img_id = str(self.data.iloc[idx, 0])
        # set path according to the root of mounted disk in your VM instance or local machine
        nifti_path = self.vm_path + img_id + '_3T_Structural_unproc_3T_T1w_MPR1.nii'
        mat_path = self.vm_path + img_id + '_3T_Structural_unproc_3T_T1w_MPR1.nii_4LR.mat'
        # read nii file, and change it to numpy
        nii = nib.load(nifti_path)
        image_hr = np.array(nii.dataobj)
        # read mat file, and change it to numpy
        mat = scipy.io.loadmat(mat_path)
        image_lr = mat['out_final']
        if (self.is_transform):
            sample_lr = self.transform(image_lr)
            sample_hr = self.transform(image_hr)

        return sample_lr, sample_hr

    def __len__(self):
        return len(self.data)