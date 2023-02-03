import argparse
import os
import numpy as np

import torch

from tqdm import tqdm

from model.ZSSGAN import ZSSGAN

import shutil
import json
import pickle
import copy

from utils.file_utils import copytree, save_images, save_paper_image_grid
from utils.training_utils import mixing_noise

from options.train_options import TrainOptions

from pytorch_lightning.utilities.seed import seed_everything
from myutils import *

import pytorch_lightning

import pdb
st = pdb.set_trace

torch.autograd.set_detect_anomaly(True)

SAVE_SRC = False
SAVE_DST = True

seed_everything(42)

def train(args):

    # Set up networks, optimizers.
    print("Initializing networks...")
    net = ZSSGAN(args)

    z_dim = 64 if args.sgxl else 512

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) 

    if args.diffusion: args.lr = args.lr/2.0

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    fixed_z = torch.randn(1, z_dim, device=device)
    
    #use pre-generated fixed noises to visualize results for the convinience of comparing between experiments
    z_noises = torch.load('z_noises.pt')
    crt = 0

    pbar = tqdm(range(args.iter))

    for i in pbar:
        net.train()

        sample_z = mixing_noise(args.batch, z_dim, args.mixing, device)

        #use diffusion guidance (our paper)
        if args.diffusion:
            net.zero_grad()
            [sampled_src, sampled_dst], loss = net(sample_z)
            pbar.set_description(f"Diffusion loss: {loss}")
        
        #use CLIP guidance (stylegan-fusion)
        else:
            [sampled_src, sampled_dst], loss = net(sample_z)
            pbar.set_description(f"Clip loss: {loss}")
            net.zero_grad()
            loss.backward()

        g_optim.step()

        
        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                [sampled_src_fix, sampled_dst_fix], loss = net([fixed_z], truncation=args.sample_truncation,evaluate=True)
                
                #crop images for "Car" experiments
                if args.crop_for_cars:
                    sampled_dst_fix = sampled_dst_fix[:, :, 64:448, :]

                z_monitor = z_noises[[*z_noises][crt%len([*z_noises])]].to(device)
                crt += 1
                [sampled_src_monitor, sampled_dst_monitor], loss = net([z_monitor], truncation=args.sample_truncation,evaluate=True)
                if args.crop_for_cars:
                    sampled_dst_monitor = sampled_dst_monitor[:, :, 64:448, :]

                save_images(torch.cat([sampled_src_monitor,sampled_src_fix,sampled_src,sampled_dst_monitor,sampled_dst_fix,sampled_dst],dim=0), sample_dir, "iter", 3, i)
        

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):

            if args.sg3 or args.sgxl:

                snapshot_data = {'G_ema': copy.deepcopy(net.generator_trainable.generator).eval().requires_grad_(False).cpu()}
                snapshot_pkl = f'{ckpt_dir}/{str(i).zfill(6)}.pkl'

                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

            else:
                torch.save(
                    {
                        "g_ema": net.generator_trainable.generator.state_dict(),
                    },
                    f"{ckpt_dir}/{str(i).zfill(6)}.pt",
                )
                
    # Generate 2000 images for FID score
    if True:
        for j in tqdm(range(2000)):
            z = torch.randn(1, z_dim, device=device)
            
            [sampled_src_monitor, sampled_dst_monitor], loss = net([z], truncation=args.sample_truncation,evaluate=True)
    
            individual_path = os.path.join(sample_dir,'individual')
            individual_path_frozen = os.path.join(individual_path,'frozen')
            individual_path_train = os.path.join(individual_path,'train')
            os.makedirs(individual_path_frozen,exist_ok=True)
            os.makedirs(individual_path_train,exist_ok=True)

            save_images(sampled_src_monitor, individual_path_frozen, "sample", 1, j)
            save_images(sampled_dst_monitor, individual_path_train, "sample", 1, j)

    

if __name__ == "__main__":
    device = "cuda"

    args = TrainOptions().parse()

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    auto_saveCode(args.output_dir)
    
    with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args)
    