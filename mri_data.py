"""
To generate img data from the raw mat file
"""
import h5py
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision import utils
SCALE = 100000
import random


def normalize_complex(data, eps=0.):
    if len(data.shape) == 2:
        mag = np.abs(data)
        mag_std = mag.std()
        return data / (mag_std + eps)
    elif len(data.shape) == 3:
        processed_data = []
        for i in range(data.shape[0]):
            mag = np.abs(data[i])
            mag_std = mag.std()
            data_i = data[i] / (mag_std + eps)
            processed_data.append(data_i)
    return np.stack(processed_data, axis=0)

def get_all_sample(dir_path,patient_list,mask_num=3):
    sample_list=[]
    key='kspace_single_full'
    data_path=dir_path/'FullSample'
    for patient in patient_list:
        patient_path=data_path/patient
        for file in patient_path.iterdir():
            if 'mask' in file.name:
                continue
            data=h5py.File(str(file))[key]
            t = data.shape[0]
            z = data.shape[1]
            fix_sublist=[file]  
            for i in range(t):
                for j in range(z):
                    for n in range(mask_num):
                        sample=fix_sublist+[n,i,j]
                        sample_list.append(sample)
    print(len(sample_list))
    return sample_list

class MRIdata(Dataset):
    def __init__(self, dir, patient_list, transform=None,mask_num=3):
        super(MRIdata, self).__init__()
        self.patient_list = patient_list
        self.dataroot = Path(dir)
        self.Heartregion_root = Path(dir) / 'HeartRegion'
        self.Noisepath = Path(dir) / 'Noise_Mask_new2'
        self.samplelist = get_all_sample(self.dataroot, self.patient_list, mask_num=mask_num)
        self.sample_num = len(self.samplelist)
        self.out_transform = transform

    def standardize(self,inputs):
        if len(inputs.shape) == 3:
            max_val = abs(inputs).max()
            min_val = abs(inputs).min()
            output = abs((inputs - min_val) / (max_val - min_val))
            
        elif len(inputs.shape)==4:
            output = torch.zeros_like(inputs)
            for t in range(inputs.shape[0]):
                t_max_val = torch.abs(inputs[t]).max()
                t_min_val = torch.abs(inputs[t]).min()
                output[t] = (inputs[t] - t_min_val) / (t_max_val - t_min_val)
        else:
            print('shape wrong!')
        
        return output

    def preprocess(self, img):
        normalize_img = normalize_complex(img)
        two_ch_img = np.stack((np.real(normalize_img), np.imag(normalize_img)), axis=1)
        return torch.from_numpy(two_ch_img)

    def crop_img(self, kspace, center):
        kspace = normalize_complex(kspace)
        img = np.fft.ifft2(kspace, axes=(-2, -1))
        img = np.fft.fftshift(img, axes=(-2, -1))
        img = normalize_complex(img)
        crop_img = img[:, center[0] - 64:center[0] + 64, center[1] - 64:center[1] + 64]
        crop_kspace = np.fft.fft2(crop_img, axes=(-2, -1))
        return crop_img, crop_kspace

    def corrupt_img(self,  kspace, mask, frame_list):
        assert kspace.shape[0] == len(frame_list), "kspace and frame_list must have compatible dimensions"
        assert kspace.shape[1:] == mask.shape,"kspace and mask must have compatible dimensions"
        corrupt_kspace = np.zeros_like(kspace)
        for t in range(kspace.shape[0]):
            original_frame = t
            replace_frame = frame_list[t,-1]
            assert replace_frame != original_frame,"replace_frame must be different from original_frame"
            corrupt_kspace[t] = kspace[original_frame] * mask + kspace[replace_frame] * (1 - mask)
        corrupt_img = np.fft.ifft2(corrupt_kspace,axes=(-2,-1))
        return corrupt_img ,corrupt_kspace
    def get_kspace(self,img):
        img_np = img.numpy()
        complex_img = img_np[:, 0, :, :] + 1j * img_np[:, 1, :, :]
        kspace = np.fft.fft2(complex_img, axes=(-2, -1))
        two_ch_kspace=np.stack((np.real(kspace), np.imag(kspace)), axis=1)
        return kspace,two_ch_kspace
       

    def __getitem__(self, index):
        local_index = self.samplelist[index]
        z = int(local_index[-1])
        t = int(local_index[-2])
        n = int(local_index[-3])
        file_path = local_index[0]
        patient = file_path.parent.name
        file = file_path.stem
        slice_file = file + f'_{z}.npy'
        slice_path = self.dataroot / 'FullSample_slice' / patient / slice_file
        k_grth_all = np.load(str(slice_path))
        region_file = file + '_heartregion.npy'
        reigion_path = self.Heartregion_root / patient / region_file
        center = np.load(reigion_path)
        noise_file = file + f'_{z}_noise.npz'
        noise_path = self.Noisepath / patient / noise_file
        mask = np.load(noise_path)['mask'][n]
        frame_list = np.load(noise_path)['frame'][n]
        img, kspace = self.crop_img(k_grth_all, center)
        all_corrupt_img, all_corrupt_kspace = self.corrupt_img( kspace, mask, frame_list)
        
        gt_image = self.preprocess(img)
        condition = self.preprocess(all_corrupt_img)
        
        std_gt_image=self.standardize(gt_image).to(torch.float32)
        std_condition_image=self.standardize(condition).to(torch.float32)
        complex_condition_kspace,real_condition_kspace=self.get_kspace(std_condition_image)
        # 应用 torchvision.transforms
        if self.out_transform is not None:
            data=torch.cat((std_gt_image, std_condition_image), axis=1)
            transform_data= self.out_transform(data)
            transformed_image=transform_data[:,:2]
            transformed_condition=transform_data[:,2:]
        

        image = transformed_image[t]
        condition =transformed_condition[t]
        conditioned_real_k=real_condition_kspace[t]
        conditioned_cplx_k=complex_condition_kspace[t]
        path=f"{patient}/{file}_{z}_{t}_{n}"
       
        
        output = {"input": condition, "GT": image, "condtioned_kspace":conditioned_real_k,"mask":mask,"ipath":path}

        return output
    
    def __len__(self):
        return self.sample_num
