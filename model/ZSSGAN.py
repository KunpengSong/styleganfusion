import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import torch
import torchvision.transforms as transforms

import numpy as np
import copy
import pickle

from functools import partial

from model.sg2_model import Generator, Discriminator
from criteria.clip_loss import CLIPLoss       
import legacy as legacy

from sd import StableDiffusion as StableDiffusion_loss_model
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F

from myutils import *
from load_EG3D import EG3D

import pdb
st = pdb.set_trace

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Generator, self).__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

        
    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers() 
        else: 
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

    def trainable_params(self):
        params = []
        for layer in self.get_training_layers():
            params.extend(layer.parameters())

        return params

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

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    #TODO Maybe convert to kwargs
    def forward(self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True):
        img = self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code)
        img = F.interpolate(img[0], (512, 512), mode='bilinear', align_corners=False)
        return [img.clamp(-1,1),None]


class ZSSGAN(torch.nn.Module):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = 'cuda:0'

        # fine-tune a EG3D model
        if self.args.EG3D:
            self.generator_frozen = EG3D(self.args)
            self.generator_frozen.freeze_layers()

            self.generator_trainable = EG3D(self.args)
            self.generator_trainable.train()

            # stylegan-fusion
            if self.args.diffusion:
                self.generator_trainable.freeze_layers()
                requires_grad(self.generator_trainable.generator.backbone.synthesis,flag=True)
                requires_grad(self.generator_trainable.generator.superresolution,flag=True)
                if self.args.freeze_bias: self.generator_trainable.freeze_bias()
            
            # stylegan-nada
            else:
                requires_grad(self.generator_trainable.generator.backbone.synthesis,flag=True)
            
        # fine-tune a 2D stylegan model
        else:
            self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size).to(self.device)
            self.generator_frozen.freeze_layers()
            self.generator_frozen.eval()
            
            self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size).to(self.device)
            self.generator_trainable.freeze_layers()
            
            # stylegan-fusion (freeze all bias parameters)
            if self.args.diffusion:
                self.generator_trainable.freeze_layers()
                for n,m in self.generator_trainable.generator.input.named_modules():
                    if is_weight_module(m, n):
                        for p in m.parameters():
                            p.requires_grad = True
                for n,m in self.generator_trainable.generator.convs.named_modules():
                    if is_weight_module(m, n):
                        for p in m.parameters():
                            p.requires_grad = True
                if self.args.freeze_bias: self.generator_trainable.freeze_bias()
            
            # stylegan-nada
            else:
                self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
            
            self.generator_trainable.train()


        # Losses (stylegan-nada: CLIP guidance loss; stylegan-fusion: StableDiffusion guidance loss)
        if args.diffusion:
            self.StableDiffusion_loss_model = StableDiffusion_loss_model(args)
            if not self.args.Disable_lpip: # use LPIPs
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(self.device)
        else:
            self.clip_loss_models = {model_name: CLIPLoss(self.device, 
                                                        lambda_direction=args.lambda_direction, 
                                                        lambda_patch=args.lambda_patch, 
                                                        lambda_global=args.lambda_global, 
                                                        lambda_manifold=args.lambda_manifold, 
                                                        lambda_texture=args.lambda_texture,
                                                        clip_model=model_name) 
                                    for model_name in args.clip_models}
            self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}


        # stylegan-nada takes in source_class and target_class, stylegan-fusion only takes in target_class
        self.source_class = args.source_class
        self.target_class = args.target_class

        # used in stylegan-nada only
        if args.target_img_list is not None:
            self.set_img2img_direction()
        
        # for layer selection
        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        self.save_chosen_layer_idx = []



    def set_img2img_direction(self):
        '''
        used in stylegan-nada only
        '''
        with torch.no_grad():
            z_dim    = 64 if self.args.sgxl else 512
            sample_z = torch.randn(self.args.img2img_batch, z_dim, device=self.device)

            if self.args.sg3 or self.args.sgxl:
                generated = self.generator_trainable(self.generator_frozen.style([sample_z]))[0]
            else:
                generated = self.generator_trainable([sample_z])[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)

                model.target_direction = direction


    def determine_opt_layers(self,args,t=None):
        '''
        layer selection
        '''
        z_dim  =  512
        sample_z = torch.randn(self.args.auto_layer_batch, z_dim, device=self.device)

        initial_w_codes = self.generator_frozen.style([sample_z])
        initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.n_latent, 1)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            w_codes_for_gen = w_codes.unsqueeze(0)
            generated_from_w = self.generator_trainable(w_codes_for_gen, input_is_latent=True)[0]
            w_optim.zero_grad()
            
            # stylegan-nada
            if args.Layer_selection_CLIP:
                w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
                w_loss = torch.sum(torch.stack(w_loss))
                w_loss.backward()
            
            # stylegan-fusion
            if args.Layer_selection_Diffusion:
                frozen_generated_from_w = self.generator_frozen(w_codes_for_gen, input_is_latent=True, truncation=1.0, randomize_noise=True)[0]
                if self.args.diffusion:
                    a, b, c = self.get_diffusion_guidance(frozen_generated_from_w, generated_from_w,eval=False,select_layer=True,t=t) #(backward included)
                        
            w_optim.step()
        
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)

        self.auto_layer_k = min(self.auto_layer_k, w_codes.shape[1])
        chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())
        conv_layers = list(all_layers[4])
        idx_to_layer = all_layers[2:4] + conv_layers # add initial convs to optimization

        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx] 

        return chosen_layers, chosen_layer_idx, layer_weights.detach().cpu().numpy()

    def get_diffusion_guidance(self,
        img_org,
        img,
        eval=False,
        select_layer=False,
        t=None):
        '''
        used in stylegan-fusion only
        '''
        if eval: return 0,0,0

        [latents] = self.StableDiffusion_loss_model.to_latent([img]) # encode img to latent
        grad, loss_item = self.StableDiffusion_loss_model.train_step(latents,t=t) # get diffusion guidance loss

        
        # clamp image grad for stable training
        if not self.args.Disable_ImageGradClamp:
            x_var = img.detach().clone()
            x_var.requires_grad = True
            [latents_x] = self.StableDiffusion_loss_model.to_latent([x_var])
            latents_x.backward(grad)
            grad_x = x_var.grad.data #1,c,h,w
            grad_quantile = torch.quantile(grad_x.view(img.shape[0], -1).abs(), 0.95, dim=1)
            grad_x = torch.min(torch.max(grad_x, -grad_quantile[:,None,None,None]), grad_quantile[:,None,None,None])
            img.backward(grad_x, retain_graph=True)
        else:
            latents.backward(grad,retain_graph=True)
        
        if select_layer: 
            # when selecting layer (layer selection), only use base loss above, don't calculate directional/reconstruction loss 
            return 0,0,0
        
        # calculate directional/reconstruction loss 
        directional_loss_item = 0
        if self.args.Directional_loss == 1 or self.args.Directional_loss == 2 :
            latents_crt, latents_org = self.StableDiffusion_loss_model.to_latent([img,img_org])
            grad, directional_loss_item = self.StableDiffusion_loss_model.train_directionalLoss_step(latents_crt,latents_org,t=t) #use the same t 
            latents_crt.backward(grad,retain_graph=True)

        # lpips loss (optional)
        if self.args.Disable_lpip:
            lpips_loss = 0
        else:
            lpips_loss = self.lpips(img_org,img)
            lpips_loss = lpips_loss * 2000 * self.args.lpip_weight_multiplicity
            if self.args.EG3D: lpips_loss = lpips_loss/100
            lpips_loss.backward()
            lpips_loss = lpips_loss.item()
            
        return loss_item, directional_loss_item, lpips_loss

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=False,
        evaluate=False,
    ):

        if self.args.diffusion:
            '''
            sample a timestep t
            use the same t for base diffusion loss, directional/reconstructional loss and layer selection 
            '''
            t = torch.randint(self.args.min_step, self.args.max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = None

        if self.args.Layer_selection_CLIP or self.args.Layer_selection_Diffusion:
            if self.training and self.auto_layer_iters > 0:
                self.generator_trainable.unfreeze_layers()
                if self.args.EG3D:
                    # not supported for 3D model
                    pass
                else:
                    train_layers, chosen_layer_idx, layer_weights = self.determine_opt_layers(self.args, t=t)

                if self.args.diffusion: self.save_chosen_layer_idx.append([t.clone().cpu().numpy(),chosen_layer_idx,layer_weights])

                if not isinstance(train_layers, list):
                    train_layers = [train_layers]
                self.generator_trainable.freeze_layers()    
                self.generator_trainable.unfreeze_layers(train_layers)        
                
                if self.args.freeze_bias: self.generator_trainable.freeze_bias()
            self.zero_grad()

        # 3D
        if self.args.EG3D:

            # sample additional parameter: camera pose c
            c = self.generator_trainable.sample_camera_pose()
            with torch.no_grad():
                frozen_img = self.generator_frozen.generator(styles[0], c)['image']  
            trainable_img = self.generator_trainable.generator(styles[0], c)['image']  
            
            frozen_img = F.interpolate(frozen_img, (512, 512), mode='bilinear', align_corners=False).clamp(-1,1)
            trainable_img = F.interpolate(trainable_img, (512, 512), mode='bilinear', align_corners=False).clamp(-1,1)

            if self.args.diffusion:
                # stylegan-fusion
                try:
                    loss_item, directional_loss_item, lpips_loss = self.get_diffusion_guidance(frozen_img, trainable_img,eval=evaluate, t=t)
                    return [frozen_img, trainable_img], loss_item
                except:
                    return [frozen_img, trainable_img], 0
            else:
                # stylegan-nada
                clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))
                return [frozen_img, trainable_img], clip_loss
        
        # 2D
        else:
            with torch.no_grad():
                if input_is_latent:
                    w_styles = styles
                else:
                    w_styles = self.generator_frozen.style(styles)
                frozen_img = self.generator_frozen(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]
            trainable_img = self.generator_trainable(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]
            
            if self.args.diffusion:
                # stylegan-fusion
                loss_item, directional_loss_item, lpips_loss = self.get_diffusion_guidance(frozen_img, trainable_img,eval=evaluate, t=t)
                return [frozen_img, trainable_img], loss_item
            else:
                # stylegan-nada
                clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))
                return [frozen_img, trainable_img], clip_loss


    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())
        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
