#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 14:20:49 2023

@author: x4nno
"""


# adapted from https://github.com/sksq96/pytorch-vae/blob/master/vae-cnn.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import sys
# from torchsummary import summary

# from pushover import notify
# from utils import makegif
from random import randint

from IPython.display import Image
from IPython.core.display import Image, display

from matplotlib import pyplot as plt

import pickle
import numpy as np

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device("cpu")
    print("Device set to : cpu")

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=64*49):
        return input.view(input.size(0), size, 1, 1)
    
class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=64*49, z_dim=4):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, image_channels, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(32, image_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

def get_model(path_to_model_dict, device="cuda"):
    model = VAE(image_channels=1).to(device)
    
    model.load_state_dict(torch.load(path_to_model_dict))
    model.eval()
    
    return model


def train_vae(e2e_traj_dir, image_channels, h_dim, z_dim):
    with open(e2e_traj_dir, "rb") as f:
        e2e_trajs = pickle.load(f)
        
    
    
    all_Os = []
    for traj in e2e_trajs:
        for O, A in traj:
            all_Os.append(O)
    
    all_Os = np.array(all_Os)
    
    vae = VAE(image_channels=image_channels, h_dim=h_dim, z_dim=z_dim).to(device)
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    print("hold ")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    traj_dir = "/home/x4nno/Documents/PhD/FRACOs_a/trajectories/e2e_traj/atari:BreakoutNoFrameskip-v4/trajs.p"
    
    train_vae(traj_dir, 4, 64*64, 20)
    
    bs = 1024
    
    # create our data
    SCALE = False
    VERBOSE = False
    CHAIN = False
    
    load_dataset = True
    load_model = True
    
    sys.path.append("/home/x4nno/Documents/PhD/FRACOs_v3/utils")
    from VAE_creation import generate_VAE_training_environments, custom_dataset
    
    if not load_dataset:
    
        _, train_dataset_pre = generate_VAE_training_environments(number_of_environments=10240,
                                                                  flat=False, scale=SCALE,
                                                                  remove_agent=True,
                                                                  remove_goal=True,
                                                                  chain_stack=CHAIN)
        
        print("created train datasets")
        
        _, test_dataset_pre = generate_VAE_training_environments(number_of_environments=1024,
                                                                  flat=False, scale=SCALE,
                                                                  remove_agent=True,
                                                                  remove_goal=True,
                                                                  chain_stack=CHAIN)
        
        print("created test datasets")
        
        pickle.dump(train_dataset_pre, open("vae_train_dataset_pre.p", "wb"))
        pickle.dump(test_dataset_pre, open("vae_test_dataset_pre.p", "wb"))
        
    else:
        train_dataset_pre = pickle.load(open("vae_train_dataset_pre.p", "rb"))
        test_dataset_pre = pickle.load(open("vae_test_dataset_pre.p", "rb"))
        
    train_dataset_pre = np.asarray(train_dataset_pre)
    train_dataset = custom_dataset(train_dataset_pre)
    test_dataset = custom_dataset(test_dataset_pre)
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
    
    fixed_x, _ = next(iter(dataloader))
    image_channels = fixed_x.size(1)
    model = VAE(image_channels=image_channels).to(device)
    if load_model:
        model.load_state_dict(torch.load('vae.torch', map_location=device))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    
    
    epochs = 50
    for epoch in range(epochs):
        for idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            to_print = "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(epoch+1, 
                                    epochs, loss.data.item()/bs, bce.data.item()/bs, kld.data.item()/bs)
            print(to_print)
            
        plt.imshow(images[0][0].cpu().detach().numpy())
        plt.title("original at epoch: {}".format(epoch))
        plt.show()
        plt.clf()
        
        plt.imshow(recon_images[0][0].cpu().detach().numpy())
        plt.title("recon image at epoch {}".format(epoch))
        plt.show()
        plt.clf()
    
    
        torch.save(model.state_dict(), 'vae.torch')
        
    # then run a test sample here!!