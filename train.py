import os
from mpi4py import MPI

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
from torch.utils.data.distributed import DistributedSampler

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
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from mri_data import MRIdata
from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
current_time = strftime("%m%d_%H_%M", gmtime())
current_day = strftime("%m%d", gmtime())

data_dir='/mnt/datasets/CMR/MICCAIChallenge2023/ChallengeData/SingleCoil/Cine/PD/train/TrainingSet'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logdir = "./log"
patient_list=['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010',
                      'P011', 'P012', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019', 'P020',
                      'P021', 'P022', 'P023', 'P024', 'P025', 'P026', 'P027', 'P028', 'P029', 'P030',
                      'P031', 'P032', 'P033', 'P034', 'P035', 'P036', 'P037', 'P038', 'P039', 'P040',
                      'P041', 'P042', 'P043', 'P044', 'P045', 'P046', 'P047', 'P048', 'P049', 'P050',
                      'P051', 'P052', 'P053', 'P054', 'P055', 'P056', 'P057', 'P058', 'P059', 'P060',
                      'P061', 'P062', 'P063', 'P064', 'P065', 'P066', 'P067', 'P068', 'P069', 'P070',
                      'P071', 'P072', 'P073', 'P074', 'P075', 'P076', 'P077', 'P078', 'P079', 'P080',
                      'P081', 'P082', 'P083', 'P084', 'P085', 'P086', 'P087', 'P088', 'P089', 'P090',
                      'P091', 'P092', 'P093', 'P094', 'P095', 'P096', 'P097', 'P098', 'P099', 'P100',
                      'P101','P102','P103','P104','P105','P106','P107','P108','P109','P110']

############################


def main():
    dist_util.setup_dist()
    logger.configure(dir=logdir)
    arg_dict = model_and_diffusion_defaults()
    arg_dict["image_size"]=128
    # arg_dict["num_channels"]=128
    # arg_dict["rrdb_blocks"]=2
    print(arg_dict)
    model, diffusion = create_model_and_diffusion(**arg_dict)
    logger.log("creating model and diffusion...")
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    tsfm = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
       
    ])  
    dataset = MRIdata(data_dir,patient_list, tsfm)

    train_set = dataset
    print(len(train_set))

    # 3. Create data loaders
    loader_args = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=8, **loader_args, drop_last=True)

    def load_gen(loader):
        while True:
            yield from loader
    train_gen = load_gen(train_loader)

    TrainLoop(
            model=model,
            diffusion=diffusion,
            data=train_gen,
            batch_size=8,
            microbatch=-1,
            lr=1e-5,
            ema_rate="0.9999",
            log_interval=200,
            save_interval=10000,
            resume_checkpoint="",
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=schedule_sampler,
            weight_decay=0.0,
            lr_anneal_steps=0,
            clip_denoised=False,
            logger=logger,
            image_size=128,
            val_dataset=None,
            run_without_test=True,
        ).run_loop(start_print_iter=100000)

if __name__ == "__main__":
    main()
