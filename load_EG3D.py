import os
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

from torchvision.utils import save_image


import pdb
st = pdb.set_trace

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class EG3D(nn.Module):
    def __init__(self,args):
        super(EG3D, self).__init__()
        
        self.args = args

        # if 'face' in self.args.output_dir: 
        #     ckpt = '../../eg3d-main/eg3d/files/ffhq512-128.pkl'
        # elif 'cat' in self.args.output_dir: 
        #     ckpt = '../../eg3d-main/eg3d/files/afhqcats512-128.pkl'
        ckpt = self.args.frozen_gen_ckpt
        
        with open(ckpt, 'rb') as f:
            self.generator = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
        focal_length = 4.2647
        self.intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]]).cuda()
        self.pitch_range = 0.25
        self.yaw_range = 0.35
        self.camera_lookat_point = torch.tensor(self.generator.rendering_kwargs['avg_camera_pivot'])

    def get_all_layers(self):
        return list(self.generator.children())

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)
            
    def freeze_bias(self):
        for n,p in self.named_parameters():
            if n.endswith('.bias'): p.requires_grad = False

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def sample_camera_pose(self):
        frac = np.random.rand(1)
        # frac = 0
        #cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, camera_lookat_point, radius=G.rendering_kwargs['avg_camera_radius']).cuda()
        cam2world_pose = LookAtPoseSampler.sample(3.14/2 + self.yaw_range * np.sin(2 * 3.14 * frac),3.14/2 -0.05 + self.pitch_range * np.cos(2 * 3.14 * frac),self.camera_lookat_point, radius=self.generator.rendering_kwargs['avg_camera_radius']).cuda()
        c = torch.cat([cam2world_pose.reshape(-1, 16), self.intrinsics.reshape(-1, 9)], 1).cuda() 
        return c

    def style(self, styles, c):
        '''
        Convert z codes to w codes.
        '''
        ws = [self.generator.mapping(s, c) for s in styles]
        return ws
    
    def forward(self,w,c):
        img = self.generator.synthesis(w, c)['image']  
        return img

    def generate_one(self):
        z = torch.randn([1, self.generator.z_dim]).cuda()    # latent codes
        
        c = self.sample_camera_pose()
        
        img = self.generator(z, c)['image']  
        # save_image((img+1)/2.0,f'{i}.png')
        return img
