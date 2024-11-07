#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 10:04:28 2023

@author: x4nno
"""

import torch
import torch.nn as nn

import numpy as np

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys 
import torch.nn.functional as F
import torch.optim as optim

import pickle

sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")

from sklearn.manifold import TSNE

class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3,
                 hidden_dim4, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim1)
        self.FC_input2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.FC_input3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.FC_input4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.FC_mean  = nn.Linear(hidden_dim4, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim4, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        h_       = self.LeakyReLU(self.FC_input3(h_))
        h_       = self.LeakyReLU(self.FC_input4(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim1, hidden_dim2, hidden_dim3,
                 hidden_dim4, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim4)
        self.FC_hidden2 = nn.Linear(hidden_dim4, hidden_dim3)
        self.FC_hidden3 = nn.Linear(hidden_dim3, hidden_dim2)
        self.FC_hidden4 = nn.Linear(hidden_dim2, hidden_dim1)
        self.FC_output = nn.Linear(hidden_dim1, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        h     = self.LeakyReLU(self.FC_hidden3(h))
        h     = self.LeakyReLU(self.FC_hidden4(h))
        
        x_hat = torch.sigmoid(self.FC_output(h)) # the sigmoid here is placing between 0 and 1
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
    
class Encoder_chain(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim):
        super(Encoder_chain, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim1)
        self.FC_input2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.FC_input3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.FC_mean  = nn.Linear(hidden_dim3, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim3, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        h_       = self.LeakyReLU(self.FC_input3(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var
    
class Decoder_chain(nn.Module):
    def __init__(self, latent_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(Decoder_chain, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim3)
        self.FC_hidden2 = nn.Linear(hidden_dim3, hidden_dim2)
        self.FC_hidden3 = nn.Linear(hidden_dim2, hidden_dim1)
        self.FC_output = nn.Linear(hidden_dim1, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        h     = self.LeakyReLU(self.FC_hidden3(h))
        
        x_hat = torch.sigmoid(self.FC_output(h)) # the sigmoid here is placing between 0 and 1
        return x_hat
    
class Model_chain(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model_chain, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') # can change to mean
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
    

def show_image(x, idx, scale=False, scaler=None, chain=False):
    x = x.cpu().numpy()
    if scaler != None:
        x = scaler.inverse_transform(x)
    elif scale:
        x = (x*4.0) # reverse of what we scale in the datacreation
    
    if chain:
        x = x.reshape(test_batch_size, 7, 21)
    else:
        x = x.reshape(test_batch_size, 7, 7)
    
    x = x[idx]
    
    #print(x)

    fig = plt.figure()

    plt.imshow(x)
    plt.show()
    
def save_image_custom(x, idx, name, scale=False, scaler=None, chain=False):
    x = x.cpu().numpy()
    if scaler != None:
        x = scaler.inverse_transform(x)
    elif scale:
        x = (x*4.0) # reverse of what we scale in the datacreation
    
    if chain:
        x.reshape(test_batch_size, 7, 21)
    
    else:
        x = x.reshape(test_batch_size, 7, 7)
    
    x = x[idx]
    
    #print(x)

    fig = plt.figure()
    plt.imshow(x)
    plt.savefig("pretrained/VAE/images{}.png".format(name))
    
class custom_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), torch.Tensor(self.data[idx])
    
def generate_VAE_training_environments(number_of_environments=100, 
                                       variations_in_env=10,
                                       action_list=[('up',), ('down',), ('left',), ('right',)],
                                       chain_length=3,
                                       flat=False,
                                       scale=False,
                                       add_2=False,
                                       remove_agent=False,
                                       remove_goal=False,
                                       chain_stack=False):
    # How do we chain random training environments ... 
    # Erm .... 
    
    # 1 generate random environment
    # 2 random starting and random reward 
    # 3 save observation space
    # 4 take 2 random actions, save observation spaces
    # 5 chain them together and save into a list
    
    vae_state_train_list = []
    vae_obs_train_list= []
    
    
    for env in range(number_of_environments):
        test_env = Environment(2, 3, [21, 21])
        for var in range(variations_in_env):
            
            test_env.reset(start_same=False, total_random=True)
            
            first = True
            obs_chain = []
            state_chain = []
            
            for i in range(chain_length):
                if first:
                    first=False
                else:
                    choice_index = np.random.choice(range(len(action_list)))
                    choice = action_list[choice_index]
                    # print(choice)
                    # plt.imshow(test_env.domain)
                    # plt.show()
                    test_env.step(choice)
                    
                if test_env.done:
                    break
                
                obs_temp = test_env.get_observation_space()[0]
                obs_temp = np.expand_dims(obs_temp, axis=0)
                state_temp = test_env.domain
                
                if remove_agent:
                    obs_temp[0][3][3] = 0
                    
                if remove_goal:
                    if np.where(obs_temp[0]==3)[0].size != 0:
                        goal_x = np.where(obs_temp==3)[0][0]
                        goal_y = np.where(obs_temp==3)[1][0]
                        obs_temp[goal_x][goal_y] = 0
                        
                    # In the goal also remove the wall.
                    if np.where(obs_temp==4)[0].size != 0:
                        goal_x = np.where(obs_temp[0]==4)[0][0]
                        goal_y = np.where(obs_temp[0]==4)[1][0]
                        obs_temp[0][goal_x][goal_y] = 0
                
                if add_2:
                    obs_temp = obs_temp+2
                    state_temp = state_temp+2

                if flat:
                    obs_temp[0] = obs_temp[0].reshape(7*7)
                    state_temp = state_temp.reshape((21+2)*(21+2))              
                    
                if scale:
                    obs_temp = (obs_temp)/4.0 
                    state_temp = (state_temp)/4.0 
                    
                if chain_stack:
                    obs_chain.append(obs_temp)
                    state_chain.append(state_temp)
                    
                else:
                    vae_obs_train_list.append(obs_temp)
                    vae_state_train_list.append(state_temp)
            
            if (chain_stack) & (not test_env.done):
                    vae_obs_train_list.append(np.hstack([obs_chain[0], obs_chain[1], obs_chain[2]]))
                    vae_state_train_list.append(np.hstack([state_chain[0], state_chain[1], state_chain[2]]))
    
    return vae_state_train_list, vae_obs_train_list


def get_model(path_to_model_dict, DEVICE="cuda"):
    x_dim  = 49
    hidden_dim1 = 48
    hidden_dim2 = 32
    hidden_dim3 = 16
    latent_dim = 8
        
    encoder = Encoder(x_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim=latent_dim)
    decoder = Decoder(output_dim=x_dim, hidden_dim1=hidden_dim1,\
                      hidden_dim2=hidden_dim2,\
                          hidden_dim3 = hidden_dim3, latent_dim=latent_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    
    model.load_state_dict(torch.load(path_to_model_dict))
    model.eval()
    
    return model, encoder, decoder


def transform_obs_0_vae(all_obs_0_space, encoder, decoder, VERBOSE=False):
    all_obs_0_latents = []
    
    for obs_0 in all_obs_0_space:
        if VERBOSE:
            plt.imshow(obs_0)
            plt.show()
        obs_0 = obs_0.reshape(1, 7*7)
        obs_temp = torch.tensor(obs_0).to(DEVICE)
        
        obs_latent = encoder(obs_temp.float()) # it is already a float? why does this work ..
        
        all_obs_0_latents.append(obs_latent[0][0].detach().cpu().numpy()) # I hate this syntax, who said python was easy ... 
        
        if VERBOSE:
            print(obs_latent)
 
            obs_hat = decoder(obs_latent[0]) 
            obs_hat = obs_hat[0].cpu().detach().numpy()
            
            obs_hat = obs_hat.reshape(7,7)
            plt.imshow(obs_hat)
            plt.show()
    
    return all_obs_0_latents


if __name__ == "__main__":

    
    dataset_path = '~/datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")

    train_batch_size = 1024
    test_batch_size = 512

    x_dim  = 49
    hidden_dim1 = 40
    hidden_dim2 = 32
    hidden_dim3 = 24
    hidden_dim4 = 16
    latent_dim = 8
    
    x_dim_chain  = 147
    hidden_dim1_chain = 128
    hidden_dim2_chain = 64
    hidden_dim3_chain = 32
    latent_dim_chain = 16

    lr = 1e-3

    epochs = 10

    SCALE = False
    VERBOSE = False
    CHAIN = False
    
    load_dataset = False
    

    kwargs = {'num_workers': 1, 'pin_memory': True} 
    
    if not load_dataset:
    
        _, train_dataset_pre = generate_VAE_training_environments(number_of_environments=300,
                                                                  flat=True, scale=SCALE,
                                                                  remove_agent=True,
                                                                  remove_goal=True,
                                                                  chain_stack=CHAIN)
        
        print("created train datasets")
        
        _, test_dataset_pre = generate_VAE_training_environments(number_of_environments=100,
                                                                  flat=True, scale=SCALE,
                                                                  remove_agent=True,
                                                                  remove_goal=True,
                                                                  chain_stack=CHAIN)
        
        print("created test datasets")
        
        pickle.dump(train_dataset_pre, open("vae_train_dataset_pre.p", "wb"))
        pickle.dump(test_dataset_pre, open("vae_test_dataset_pre.p", "wb"))
        
    else:
        train_dataset_pre = pickle.load(open("vae_train_dataset_pre.p", "rb"))
        test_dataset_pre = pickle.load(open("vae_test_dataset_pre.p", "rb"))
    
    # scaler = StandardScaler()
    # train_dataset = scaler.fit_transform(train_dataset)
    # test_dataset = scaler.transform(test_dataset)
    
    
    train_dataset = custom_dataset(train_dataset_pre)
        
    test_dataset = custom_dataset(test_dataset_pre)
    
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=test_batch_size, shuffle=False, **kwargs)
    
    if CHAIN:
        encoder = Encoder_chain(x_dim_chain, hidden_dim1_chain, hidden_dim2_chain, hidden_dim3_chain, latent_dim=latent_dim_chain)
        decoder = Decoder_chain(output_dim=x_dim_chain, hidden_dim1=hidden_dim1_chain,\
                          hidden_dim2=hidden_dim2_chain,\
                              hidden_dim3 = hidden_dim3_chain, latent_dim=latent_dim_chain)
    
        model = Model_chain(Encoder=encoder, Decoder=decoder).to(DEVICE)
    
    else:
        encoder = Encoder(x_dim, hidden_dim1, hidden_dim2, hidden_dim3,
                          hidden_dim4, latent_dim=latent_dim)
        decoder = Decoder(output_dim=x_dim, hidden_dim1=hidden_dim1,\
                          hidden_dim2=hidden_dim2,\
                              hidden_dim3 = hidden_dim3,
                              hidden_dim4 = hidden_dim4, latent_dim=latent_dim)
    
        model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=lr)
    
    print("Start training VAE...")
    model.train()
    
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader): # Why is the last one not at the batch size?
            #print(x.shape[0])
            if x.shape[0] != train_batch_size:
                break
            
            if CHAIN:
                x = x.view(train_batch_size, x_dim_chain)
            else:
                x = x.view(train_batch_size, x_dim)
                
            x = x.to(DEVICE)
    
            optimizer.zero_grad()
    
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
       
        if VERBOSE:
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)
            
            h = x[0].cpu().detach().numpy()
            h_hat = x_hat[0].cpu().detach().numpy()
            
            if CHAIN:
                h = h.reshape(7,21)
                h_hat = h_hat.reshape(7,21)
            else:
                h = h.reshape(7,7)
                h_hat = h_hat.reshape(7,7)
            
            if SCALE:
                h = h*4
                h_hat = h_hat*4
    
            plt.imshow(h)
            plt.show()
            plt.imshow(h_hat)
            plt.show()
        
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*train_batch_size))
        
    print("Finish!!")
    
    torch_save_dir = "pretrained/VAE"
    torch.save(model.state_dict(), torch_save_dir+"/vae_model_test.pt")
    
    model.eval()

    no_of_images = 0
    total = 10
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if CHAIN:
                x = x.view(test_batch_size, x_dim_chain)
            else:
                x = x.view(test_batch_size, x_dim)
                
            x = x.to(DEVICE)
            
            x_hat, _, _ = model(x)
            
            show_image(x, idx=0, scale=SCALE, chain=CHAIN)
            show_image(x_hat, idx=0, scale=SCALE, chain=CHAIN)
            
            save_image_custom(x, 0, "x_{}".format(no_of_images), chain=CHAIN)
            save_image_custom(x_hat, 0, "x_hat_{}".format(no_of_images), chain=CHAIN)
            
            no_of_images += 1
            if no_of_images >= total:
                break
    
    
    
    with torch.no_grad():
        noise = torch.randn(test_batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)
        
    show_image(generated_images, idx=12, chain=CHAIN)
