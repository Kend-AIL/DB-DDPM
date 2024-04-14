import os

################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
result_dir = "/mnt/datasets/CMR/MICCAIChallenge2023/real_time_result_without_kspace"
model_path = '/root/DiffCMR_new/log/ema.pt'
# val_pair_file = "/home/txiang/CMRxRecon/CMRxRecon_Repo/dataset/train_pair_file/Task2_acc_10_val_pair_file_npy_clean.txt"
val_bs = 24
################################################

from improved_diffusion import dist_util, logger
# from datasets.city import load_data, create_dataset
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.utils import set_random_seed, set_random_seed_for_iterations
import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from CMRxRecon import MRIdata
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())
def load_mri(kdata_dir,t):
    def normalize_complex(data, eps=1e-9):
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
    def standardize(inputs):
        if len(inputs.shape) == 3:
            max_val = abs(inputs).max()
            min_val = abs(inputs).min()
            output = (inputs - min_val) / (max_val - min_val)
            
        elif len(inputs.shape)==4:
            output = np.zeros_like(inputs)
            for t in range(inputs.shape[0]):
                t_max_val = np.abs(inputs[t]).max()
                t_min_val = np.abs(inputs[t]).min()
                output[t] = (inputs[t] - t_min_val) / (t_max_val - t_min_val)
        else:
            print('shape wrong!')
        
        return output
    def preprocess(img):
            normalize_img = normalize_complex(img)
            two_ch_img = np.stack((np.real(normalize_img), np.imag(normalize_img)), axis=1)
            return two_ch_img
    kdata=np.load(kdata_dir)[:22]
    img=np.fft.ifft2(kdata,axes=(-2,-1))
    kspace=np.fft.fft2(img,axes=(-2,-1))
    
    pre_img=preprocess(img)
    std_image=standardize(pre_img)
    print(std_image.max(),std_image.min())
    return std_image[t]

#inference check
from improved_diffusion.sampling_util import sample_one_withou_ksapce
dist_util.setup_dist()
logger.configure(dir=result_dir)
arg_dict = model_and_diffusion_defaults()

arg_dict["image_size"]=128
arg_dict["diffusion_steps"]=1000
print(arg_dict)
model, diffusion = create_model_and_diffusion(**arg_dict)
model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
logger.log("creating model and diffusion...")
model.to(dist_util.dev())
model.eval()
kspace_dir='/mnt/datasets/CMR/Heart_Failure_ChinaJapanHospital/Data_processed/transformed_dataset_1.npy'
t=21
inputs=abs(load_mri(kspace_dir,t))
sample_one_withou_ksapce(inputs,diffusion, model, result_dir, logger, True, vote_num=4)