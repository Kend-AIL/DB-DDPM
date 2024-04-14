import os

################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
result_dir = "/mnt/datasets/CMR/MICCAIChallenge2023/result_with_ksapce_new"
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
from MoCo_data import MRIdata
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())


#inference check
from improved_diffusion.sampling_util import sampling_with_kspace
dist_util.setup_dist()
logger.configure(dir=result_dir)
arg_dict = model_and_diffusion_defaults()

arg_dict["image_size"]=128
arg_dict["diffusion_steps"]=200
print(arg_dict)
model, diffusion = create_model_and_diffusion(**arg_dict)
model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
logger.log("creating model and diffusion...")
model.to(dist_util.dev())
model.eval()
tsfm = transforms.Compose([
        
    transforms.Resize(128),
       
    ])  
data_dir='/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
patient_list=['P111', 'P112', 'P113', 'P114', 'P115', 'P116', 'P117', 'P118', 'P119', 'P120']
dataset = MRIdata(data_dir, patient_list,tsfm)

sampling_with_kspace(val_bs, diffusion, model, result_dir, dataset, logger, True, vote_num=4)