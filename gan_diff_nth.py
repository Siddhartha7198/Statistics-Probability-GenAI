#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:49:33 2024
GAN applied on nth root of 2D diffraction intensity 
@author: poddar
"""

import os
import sys
import torch
import numpy as np
from torch import fft as ft
import torch.nn.functional as F
from dataclasses import dataclass
from torch import nn, optim, autograd 
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from torchvision.transforms import GaussianBlur as G
plt.style.use("default.pltstyle") 
# %%

###### Setting Device to GPU #########
if torch.cuda.is_available():
    device = torch.device("cuda")      
       
dtype = torch.float32

### reproducible result ###
# random_seed = 42
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

#%%

@dataclass
class Hyperparameter:
    
    ##### training #####
    batchsize: int          = 8
    num_iters: int          = 100000    
    
    ##### optimizer #####
    dis_lr: float            = 1e-3
    dis_lr_cutoff: float     = 1e-5
    
    vol_lr: float            = 1e-3
    vol_lr_cutoff: float     = 1e-5
    
    beta1: float             = 0.5
    beta2: float             = 0.9
    
    #### scheduler #####
    step: int               = 1000
    gamma: float             = 0.99
    
    ##### network ######
    n_critic: int           = 1
    critic_size: int        = 2048
    critic_hidden_size: int = 10
    gp_lambda: float         = 1
    
    ##### Poisson noise #####
    poisson_noise: int      = int(sys.argv[3])*1e5

    #### Gaussian noise #####
    gaussian: int           = int(sys.argv[1])
    noise_factor: float      = float(sys.argv[2])
        
hp = Hyperparameter()
print(hp, flush=True)
print("\n", flush=True)


# %%
    
##################### Initialize weights ###########################

def weights_init(m):
    
    if isinstance(m, nn.Conv2d):
        
        if m.weight is not None:            
            nn.init.xavier_normal_(m.weight)           
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
        
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

########################## Critic #########################
            
class Critic(nn.Module):
    
    def __init__(self):
        super(Critic, self).__init__()
        
        # Define CNN layers
        channels = [1] + [hp.critic_size // (2 ** i) for i in range(5, -1, -1)] 
        cnn_layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            cnn_layers.append(nn.Conv2d(in_ch, out_ch, 3, 2, 1))
            cnn_layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            cnn_layers.append(nn.GELU())
            # cnn_layers.append(nn.LeakyReLU(0.1))
            
        
        cnn_layers.append(nn.Flatten())
        self.cnn_net = nn.Sequential(*cnn_layers)
        
        # Define fully connected layers
        fc_layers = []
        fc_sizes = [hp.critic_size * 4, hp.critic_hidden_size, 1]
        for in_size, out_size in zip(fc_sizes[:-1], fc_sizes[1:]):
            fc_layers.append(nn.Linear(in_size, out_size))
            if out_size != 1:  
                fc_layers.append(nn.GELU())
                # fc_layers.append(nn.LeakyReLU(0.1))
        
        self.critic_net = nn.Sequential(*fc_layers)
        self.apply(weights_init)    
    
    def forward(self, image):
        return self.critic_net(self.cnn_net(image))
    
    
#%%

def noisy_images(real_images, fake_images, gaus2d, hp):
       
    if hp.poisson_noise == 0:  
                          
        if hp.gaussian == 0:
            return real_images, fake_images  # (No Poisson, No Gaussian)
        
        ##### Background Gaussian signal #####  
        else:
            real_images = real_images + hp.noise_factor * gaus2d + 0.05*torch.rand(gaus2d.shape).cuda()
            fake_images_gaussian = fake_images + hp.noise_factor * gaus2d + 0.05*torch.rand(gaus2d.shape).cuda()  # (No Poisson, with Gaussian)
            return real_images, fake_images_gaussian
    
    ##### Poisson noisy image ##### 
    else:
               
        if hp.gaussian == 0:
            real_images = (torch.poisson(hp.poisson_noise * real_images**n) / hp.poisson_noise)**(1/n)
            with torch.no_grad(): 
                poisson_noise = (torch.poisson(hp.poisson_noise * fake_images.data**n) / hp.poisson_noise)**(1/n)  - fake_images.data

            fake_images_poisson = fake_images + poisson_noise   # (With Poisson, No Gaussian)
            return real_images, fake_images_poisson
        
        ##### Background Gaussian signal #####
        else:
            real_images = real_images + hp.noise_factor * gaus2d + 0.05*torch.rand(gaus2d.shape).cuda()
            fake_images_gaussian = fake_images + hp.noise_factor * gaus2d + 0.05*torch.rand(gaus2d.shape).cuda()  # (With Poisson, with Gaussian)

            real_images = (torch.poisson(hp.poisson_noise * real_images**n) / hp.poisson_noise)**(1/n)
            with torch.no_grad(): 
                poisson_noise = (torch.poisson(hp.poisson_noise * fake_images_gaussian.data**n) / hp.poisson_noise)**(1/n) - fake_images_gaussian.data

            fake_images_poisson_gaussian = fake_images_gaussian + poisson_noise  
            return real_images, fake_images_poisson_gaussian

def normalize_image(tensor):
    # min_val = tensor.view(tensor.shape[0], -1).min(dim=1, keepdim=True)[0].view(tensor.shape[0], 1, 1, 1)
    max_val = tensor.view(tensor.shape[0], -1).max(dim=1, keepdim=True)[0].view(tensor.shape[0], 1, 1, 1)
    return  tensor/max_val #(tensor - min_val) / (max_val - min_val + 1e-6)


def crop(vol):
    size = vol.shape[-1] // 4
    center = vol.shape[-1] // 2
    return vol[center - size:center + size, center - size:center + size, center - size:center + size]

def generate_2d_gaussian(size, sigma):

    H = W = size
    y = torch.arange(0, H).unsqueeze(1).repeat(1, W)
    x = torch.arange(0, W).unsqueeze(0).repeat(H, 1)
    gauss = torch.exp(-((x - H//2) ** 2 + (y - W//2) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.max()  # normalize to have max value 1
    return gauss


def mask(data, hole_size=10):
    
    temp = torch.ones_like(data)

    dims = data.ndim  # Get number of dimensions

    if dims == 3:  # Case: (height, length, width) -> 3D volume
        ctr_h = data.shape[0] // 2
        ctr_l = data.shape[1] // 2
        ctr_w = data.shape[2] // 2
        half_hole = hole_size // 2

        temp[ctr_h - half_hole : ctr_h + half_hole + 1,
             ctr_l - half_hole : ctr_l + half_hole + 1,
             ctr_w - half_hole : ctr_w + half_hole + 1] = 0

    elif dims == 4:  # Case: (batch, channel, height, width) -> 2D images
        ctr_h = data.shape[-2] // 2
        ctr_w = data.shape[-1] // 2
        half_hole = hole_size // 2

        temp[..., ctr_h - half_hole : ctr_h + half_hole + 1,
                  ctr_w - half_hole : ctr_w + half_hole + 1] = 0

    return temp

    
def rmse(target, data):
    data = data / torch.max(data) if torch.max(data) > 1 else data
    return torch.sqrt(torch.sum((target - data) ** 2) / torch.sum(target ** 2))


def sample_euler_angles(batchsize):
    angles = 2 * torch.rand(batchsize, 3).cuda()
    angles[:, 0] *= torch.pi
    angles[:, 2] *= torch.pi
    angles[:, 1] = torch.arccos(angles[:, 1] - 1.0)
    return angles


def rot_mat(angles):
    # Calculate cosine and sine of angles
    c, s = angles.cos(), angles.sin()
    
    # Extract components of cosine and sine for each axis
    c1, c2, c3 = c[:, 0], c[:, 1], c[:, 2]
    s1, s2, s3 = s[:, 0], s[:, 1], s[:, 2]

    # Create the 3x3 rotation matrix
    rot_3x3 = torch.stack([
        torch.stack([c3 * c2 * c1 - s3 * s1, c3 * c2 * s1 + s3 * c1, -c3 * s2], dim=1),
        torch.stack([-s3 * c2 * c1 - c3 * s1, -s3 * c2 * s1 + c3 * c1, s3 * s2], dim=1),
        torch.stack([s2 * c1, s2 * s1, c2], dim=1),
    ], dim=1)

    # Add an extra column of zeros to the rotation matrix to make it 3x4
    zeros_col = torch.zeros(rot_3x3.shape[0], 3, 1, dtype=rot_3x3.dtype).cuda()
    rot_3x4 = torch.cat([rot_3x3, zeros_col], dim=-1)

    return rot_3x4


def rot_vol(vol, grid, dtype):
                     
    rotvol = F.grid_sample(vol, grid,align_corners=False)
    slices = rotvol[...,vol.shape[-1]//2]
    
    return slices
   
#%%

def train(target_vol):	    

    target_vol = target_vol.unsqueeze(0).unsqueeze(0)
    
    ####### Downsampling ######
    # target_vol = nn.AvgPool3d(3, 2, 1)(target_vol)
    
    ####### Hole at center ########
    # mask3d = mask(target_vol)     
    # target_vol = target_vol*mask3d
    
    ####### Normalization ########## 
    norm = torch.max(target_vol) 
    target_vol /= norm            
    
    ###### Volume Initialization ######    
    initial_vol = torch.zeros(target_vol.shape).cuda() + 1e-6
    initial_vol = nn.Parameter(initial_vol)

    
    ########### Background modeling ###############   
    with torch.no_grad(): gaus2d = generate_2d_gaussian(target_vol.shape[-1],target_vol.shape[-1]/8).cuda()
    
    ##############################   Training optimizer and scheduler    ##################################
    
    vol_optim = optim.Adam([{'params': initial_vol}], lr=hp.vol_lr, betas=(hp.beta1, hp.beta2))

    vol_scheduler = optim.lr_scheduler.StepLR(vol_optim, step_size=hp.step//hp.n_critic, gamma=hp.gamma)

    critic = Critic().cuda()          

    critic_optimizer = optim.Adam(critic.parameters(), lr=hp.dis_lr, betas=(hp.beta1, hp.beta2))

    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=hp.step, gamma=hp.gamma)


    error = torch.ones(hp.num_iters)
    rmse_error = torch.ones(hp.num_iters)
    
    target_vol_copied = torch.cat([target_vol.clone() for _ in range(hp.batchsize)])  
    
    ###############################  Training  #############################################
            
    for i in range(1,hp.num_iters+1):
            
        angles = sample_euler_angles(hp.batchsize)           
        rot_matx = rot_mat(angles).type(dtype) 
        grid = F.affine_grid(rot_matx, target_vol_copied.size(),align_corners=False).type(dtype)

        real_images = rot_vol(target_vol_copied, grid, dtype)    

        
        initial_vol_copied = torch.cat([initial_vol.clone() for _ in range(hp.batchsize)])  
                    
           
        angles = sample_euler_angles(hp.batchsize)        
        rot_matx = rot_mat(angles).type(dtype)         
        grid = F.affine_grid(rot_matx, initial_vol_copied.size(),align_corners=False).type(dtype)
        
        raw_fake_images = rot_vol(initial_vol_copied, grid, dtype)
        
        ##### Noisy image #####        
        real_images, fake_images = noisy_images(real_images, raw_fake_images, gaus2d, hp)
        
            
        ########################################################################################        
                    #################  Update Critic  ######################
        ########################################################################################                    
        
        critic_optimizer.zero_grad()
        
        critic_output_real = critic(real_images)
        critic_loss_real = critic_output_real.mean()

        critic_output_fake = critic(fake_images)
        critic_loss_fake = critic_output_fake.mean()

        grad_tensor = torch.ones((hp.batchsize, 1)).cuda()
        alpha = torch.rand((hp.batchsize, 1, 1, 1)).cuda()
        interpolates = (alpha * real_images + ((1. - alpha) * fake_images)).requires_grad_(True)
        d_interpolates = critic(interpolates)
        gradients = autograd.grad(d_interpolates, interpolates, grad_tensor, create_graph=True, only_inputs=True)[0]
        gradient_penalty = hp.gp_lambda * ((gradients.reshape(hp.batchsize, -1).norm(dim=1) - 1.) ** 2).mean()

        critic_loss = -critic_loss_real + critic_loss_fake  + gradient_penalty
        
        critic_loss.backward()            
        
        critic_optimizer.step()  
      
        if hp.dis_lr_cutoff < critic_scheduler.get_last_lr()[0]: critic_scheduler.step()


        #########################################################################################        
                            ############ Update Generator  #################
        #########################################################################################
        

        if i % hp.n_critic == 0:

            angles = sample_euler_angles(hp.batchsize)
            rot_matx = rot_mat(angles).type(dtype)
            grid = F.affine_grid(rot_matx, initial_vol_copied.size(),align_corners=False).type(dtype)

            raw_fake_images = rot_vol(initial_vol_copied, grid, dtype)
            
            ##### Noisy image #####        
            real_images, fake_images = noisy_images(real_images, raw_fake_images, gaus2d, hp)
            
            
            vol_optim.zero_grad()
            
            critic_output_fake = critic(fake_images)
            generator_loss = -critic_output_fake.mean() 
            
            generator_loss.backward()
            vol_optim.step()

            if hp.vol_lr_cutoff < vol_scheduler.get_last_lr()[0]: vol_scheduler.step()  


            ############### Constraint on volume ################
            
            with torch.no_grad(): 
                initial_vol.data.clamp_(min=1e-10)
                
            ########################################################                 
            
        # if i % 1000 == 0 or i == hp.num_iters:
        
        with torch.no_grad():
            recon = initial_vol.clone().detach()
            rmse_err = rmse(target_vol, recon)
            rmse_error[i-1] = rmse_err
            curr_err = -critic_loss             
            error[i-1] = curr_err
            
        print(f"Iteration: {i}/{hp.num_iters} - loss: {curr_err.cpu().item():.4f} - C(fake): {critic_loss_fake.cpu().item():.4f} - C(real): {critic_loss_real.cpu().item():.4f} - grad: {gradient_penalty.cpu().item():.4f} - rmse: {rmse_err.cpu().item():.4f}\n", flush=True)
        
        if i % 1000 == 0 or i == hp.num_iters:
            OUTPUT_PATH = './Results/'
            os.makedirs(OUTPUT_PATH, exist_ok=True)  
        
            name = f"recon_iter{i}"    
            
            with torch.no_grad():
                volume = initial_vol.clone().detach() 
                volume *= norm/torch.max(volume)
                volume = volume**n      
                volume = volume.squeeze().cpu().numpy()
                
                np.save(os.path.join(OUTPUT_PATH, name), volume)
                
            # Update best volume if the new error is the lowest
            # if curr_err < min_err:
            #     min_err = curr_err
            #     best = volume
                    
                    
        # ctr = target_vol.shape[-1]//2        
        # if i % 100 == 0: 
        #     plt.figure(figsize=(10, 6))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(target_vol[...,ctr].squeeze().detach().cpu().numpy(),norm=LogNorm(vmin=1e-1, vmax=1),cmap='jet')
        #     plt.tick_params(left = False, bottom = False,labelleft = False, labelbottom = False)
        #     plt.colorbar(shrink=0.6)
        #     plt.title("Target Slice")
            
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(initial_vol[...,ctr].squeeze().detach().cpu().numpy(),norm=LogNorm(vmin=1e-1, vmax=1),cmap='jet')
        #     plt.tick_params(left = False, bottom = False,labelleft = False, labelbottom = False)
        #     plt.colorbar(shrink=0.6)    
        #     plt.title("Reconstructed Slice")
        #     plt.suptitle(f'Iteration: {i}')
        #     plt.savefig(f'/data/finite/poddar/gan/diffgan/images/recon_iter_{i}.pdf', dpi=250)
        #     plt.close()
            
    np.save(os.path.join(OUTPUT_PATH, 'rmse_error'), rmse_error)        
    np.save(os.path.join(OUTPUT_PATH, 'recon_error'), error) 
    np.save(os.path.join(OUTPUT_PATH, 'recon_best'), volume)
        
    return volume


#%%

#### 3D density ####
image = np.load('3d_den.npy') 
image = torch.from_numpy(image).float().cuda()

####### zero padding ##########
pad = nn.ConstantPad3d(image.shape[-1]//2, 0)
image = pad(image)
        
n = 6

#### 3D diffraction image ####
image = torch.abs(ft.fftshift(ft.fftn(image)))**(2/n)       #### nth root of intensity


##### Cropping ######
if image.shape[-1] == 256:      #### ribosome #####
    image = crop(image)
else:
    image = crop(image)

 
#%%

start = timer()
recon = train(image)
stop = timer()
print(f"Time taken: {stop - start:.2f} secs")








































    
