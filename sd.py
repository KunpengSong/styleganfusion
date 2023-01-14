from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler,DDIMScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize
import numpy as np
from tqdm import tqdm
import time
from torch.nn.functional import cosine_similarity as cosine_similarity

import pdb
st = pdb.set_trace

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class StableDiffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')
        
        self.device = 'cuda'
        self.args = args
        self.prompt = args.target_class
        print(f'[INFO] loading stable diffusion...')

        if self.args.DreamBooth == 'Wavy':
            # use dreambooth Wavy checkpoint
            self.vae = AutoencoderKL.from_pretrained("wavymulder/wavyfusion", subfolder="vae", use_auth_token=self.token).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained("wavymulder/wavyfusion", subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained("wavymulder/wavyfusion", subfolder="text_encoder").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained("wavymulder/wavyfusion", subfolder="unet", use_auth_token=self.token).to(self.device)
        elif self.args.DreamBooth == "Woolitize":
            # use dreambooth Woolitize checkpoint
            self.vae = AutoencoderKL.from_pretrained("plasmo/woolitize-768sd1-5", subfolder="vae", use_auth_token=self.token).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained("plasmo/woolitize-768sd1-5", subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained("plasmo/woolitize-768sd1-5", subfolder="text_encoder").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained("plasmo/woolitize-768sd1-5", subfolder="unet", use_auth_token=self.token).to(self.device)
        else:
            # use stable-diffusion-v1-4
            self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=self.token).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
            self.unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=self.token).to(self.device)

        # 4. Create a scheduler for inference 
        self.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, skip_prk_steps=True,beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)
        self.noises = None
        print(f'[INFO] loaded stable diffusion!')
        self.text_embeddings = self.get_text_embeds([self.prompt]*1)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas)

        self.optim_noise = None

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        print(prompt)
        print(self.args.output_dir)
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def rescale(self,noise_pred):
            b,c,h,w = noise_pred.shape
            m = torch.sqrt(torch.tensor(c*h*w)).to('cuda')
            noise_pred = noise_pred*m/torch.linalg.norm(noise_pred) # no batch support
            return noise_pred

    def to_latent(self, rgbs):
        latents = []
        for rgb in rgbs:
            latents.append(self.encode_imgs(rgb))
        return latents
    
    def train_step(self, latents, t=None):
        # sample a timestep t ~ U[T_min, T_max]
        if t == None: t = torch.randint(self.args.min_step, self.args.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            noise = torch.randn_like(latents)
            latent_model_input = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latent_model_input] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=self.text_embeddings).sample

        # perform guidance (high scale from DreamFusion paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.args.CFG * (noise_pred_text - noise_pred_uncond)

        if self.args.rescaleCFG_Noise: 
            noise_pred = self.rescale(noise_pred)
            noise = self.rescale(noise)

        # clip grad for stable training?
        # if not self.args.Disable_ClampSDGrad: grad = grad.clamp(-1, 1)
        
        diff = noise_pred - noise
        w = (1 - self.alphas[t])        
        grad = w * (diff)
        
        if self.args.EG3D: grad = grad/100.0

        grad = grad.clamp(-1, 1)
        grad = torch.nan_to_num(grad)

        return grad,diff.pow(2).mean().sum().item()
    
    def train_directionalLoss_step(self, latents_crt, latents_org, t=None):
        # sample timestep t ~ U(min, max)
        if t == None: t = torch.randint(self.args.min_step, self.args.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        # Directional
        if self.args.Directional_loss == 1:
            with torch.no_grad():
                noise = torch.randn_like(latents_crt)
                latent_model_input_crt = self.scheduler.add_noise(latents_crt, noise, t)
                latent_model_input_org = self.scheduler.add_noise(latents_org, noise, t)

                text_embeddings = torch.cat([self.text_embeddings]) #U,C

                latent_model_input = torch.cat([latent_model_input_crt,latent_model_input_crt]) #CC
                noises_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond_crt, noise_pred_cond_crt = noises_pred.chunk(2)

                latent_model_input = torch.cat([latent_model_input_org,latent_model_input_org]) #OO
                noises_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                noise_pred_uncond_org, noise_pred_cond_org = noises_pred.chunk(2)

                noise_drct_crt = noise_pred_uncond_crt + self.args.CFG*(noise_pred_cond_crt - noise_pred_uncond_crt)
                noise_drct_org = noise_pred_uncond_org + self.args.CFG*(noise_pred_cond_org - noise_pred_uncond_org)
                if self.args.rescale_CFGDirectionNoise: noise_drct_crt, noise_drct_org = self.rescale(noise_drct_crt), self.rescale(noise_drct_org)
                diff = noise_drct_crt - noise_drct_org

                w = (1 - self.alphas[t])
                grad = w * (diff)

                grad = grad * self.args.directional_loss_weight_multiplicity

        # Reconstruction
        if self.args.Directional_loss == 2: 
            noise = torch.randn_like(latents_crt)
            xt = self.scheduler.add_noise(latents_crt, noise, t)  # NOTE: q_sample

            uncond_embeddings, text_embeddings = self.text_embeddings.chunk(2)
            e_t_uncond = self.unet(xt, t, encoder_hidden_states=uncond_embeddings).sample
            e_t_cond = self.unet(xt, t, encoder_hidden_states=text_embeddings).sample

            e_t  = e_t_uncond + self.args.CFG * (e_t_cond-e_t_uncond)
            e_t  = e_t.detach().clone()
            e_t.requires_grad = True

            a_t = extract_into_tensor(self.alphas, t, xt.shape)
            sqrt_one_minus_at = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
            pred_x0 = (xt - sqrt_one_minus_at * e_t) / a_t.sqrt()

            loss_recon = F.mse_loss(pred_x0, latents_org.detach())
            e_t_grad = torch.autograd.grad(loss_recon, e_t)[0]
            e_t_reg = e_t_grad.detach().clone()

            diff = self.rescale(e_t) - self.rescale(e_t_reg)

            w = (1 - self.alphas[t])
            grad = w * (diff)
            grad = grad * self.args.directional_loss_weight_multiplicity

        
        if self.args.EG3D: grad = grad/100.0

        grad = grad.clamp(-1, 1)
        grad = torch.nan_to_num(grad)

        return grad,diff.pow(2).mean().sum().item()

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.mode() * 0.18215

        return latents
