#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:19:47 2023

@author: x4nno
"""

import os
import sys
import gym
import pickle

os.getcwd()

from train.fracos_all import train_pipe
from method.fracos_PPO import fracos_PPO
from method.fracos_QL import fracos_QL
from utils.eval import create_trajectories, create_random_trajectories, create_ind_opt_trajectories
from utils.clustering import create_fractures, create_clusterer,\
get_all_s_f_index, visualize_clusters, produce_umap_reducer_pipeline, \
produce_umap_reducer_pipeline_multi
from utils.compression import cluster_PI_compression

########## Below is needed to connect MetaGridEnv to the imports ##############
## NOTE! ONLY RUN THIS CELL ONCE - IT WILL ONLY REGISTER ONCE!
import sys
sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv") # change to your location
import MetaGridEnv
from gym.envs.registration import register 

#  Register only needs to be run once (but everytime the script is run)
try:
    register( id="MetaGridEnv/metagrid-v0",
              entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")
except:
    print("MetaGridEnv is already registered")
    
####### initialize environment hyperparameters ######
    
# env_name = "CartPole-v1"
# env_name = "LunarLander-v2"
# env_name = "Breakout-ram-v4"
# env_name = "procgen:procgen-coinrun-v0"
# env_name = "MetaGridEnv"
env_name = "MetaGridEnv_Four_rooms"

###### initialize other params related to training ######

max_ep_len = 300                    # max timesteps in one episode
max_training_timesteps = int(2e6)   # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 5     # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2       # log avg reward in the interval (in num timesteps)
save_model_freq = int(1e4)      # save model frequency (in num timesteps)

eval_eps = 1                    # How many eval episodes to average over on print freq
                                # set to 0 to not do any eval (just train the policy)

MAX_DEPTH = 1                 # This is the maximum depth of clusters
                                # check that the clusters exist for each depth
                                # as will break if they don't exist
                                # see later section on how to make clusters

CURRENT_DEPTH = 0
H_level_pretrained = CURRENT_DEPTH

max_clusters_per_clusterer=0   # This will limit the maximum clusters at each depth
gen_strength=0.1                # 1 minus how strongly the clusterer should be sure that 
                                # a new point belongs to a cluster. at 0 no 
                                # inference or clusters will be used.

update_timestep = max_ep_len*5      # update policy every n timesteps

K_epochs = 200              # update policy for K epochs
eps_clip = 0.4              # clip parameter for PPO
gamma = 0.99                # discount factor

# lr_actor = 0.0003       # learning rate for actor network
# lr_critic = 0.001       # learning rate for critic network

lr_actor = 0.00002       # learning rate for actor network
lr_critic = 0.0002       # learning rate for critic network

chain_length = 0
random_seed = 0

### This below is required as MetaGridEnv needs some more params    
if env_name == "MetaGridEnv_2026":
    env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], seed=2026)
elif env_name == "MetaGridEnv":
    env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14])
elif env_name == "MetaGridEnv_Josh_grid":
    env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Josh_grid")
elif env_name == "MetaGridEnv_Four_rooms":
    env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Four_rooms")
else:
    env = gym.make(env_name)
            
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

directory = "pretrained/fracos_PPO_preTrained"
if not os.path.exists(directory):
          os.makedirs(directory)
    
directory = directory + '/' + env_name + '/'
if not os.path.exists(directory):
          os.makedirs(directory)
    
    
checkpoint_path = directory + "fracos_PPO_{}_{}_{}.pth".format(env_name, random_seed, H_level_pretrained)


# initial_state = env.env_master.domain # needed for tabular and MetaGridEnv

# init our agent.

fracos_agent = fracos_PPO(state_dim, action_dim, lr_actor,
                                      lr_critic, gamma, K_epochs, eps_clip,
                                      env_name, MAX_DEPTH, chain_length=chain_length,
                                      max_clusters_per_clusterer=max_clusters_per_clusterer,
                                      gen_strength=gen_strength, current_depth=CURRENT_DEPTH,
                                      vae_path=None #"/home/x4nno/Documents/PhD/FRACOs_v3/pretrained/VAE/MetaGridEnv/vae_cnn.torch"
                                      )

# fracos_agent = fracos_QL(state_dim, action_dim, gamma,
#                               env_name, MAX_DEPTH, initial_state, current_depth=CURRENT_DEPTH,
# #                              vae_path="/home/x4nno/Documents/PhD/FRACOs_v3/pretrained/VAE/MetaGridEnv/vae_cnn.torch"
#                              vae_path=None
#                             , max_clusters_per_clusterer=max_clusters_per_clusterer,
#                              )

# NOTE !! IF YOU ALREADY HAVE A TRAINED POLICY YOU DON@T NEED TO RUN THIS EVERYTIME!!

discrete_search_cache, env, method = train_pipe(env_name, max_ep_len, max_training_timesteps,
                                   print_freq, log_freq, save_model_freq, eval_eps,
                                   update_timestep, MAX_DEPTH, chain_length=chain_length,
                                   max_clusters_per_clusterer=max_clusters_per_clusterer,
                                   gen_strength=0.1, discrete_search_cache={},
                                                current_depth=0, # hash all below for stategen
                                    passed_env=env, reward_gen_only=True, fixed=False, 
                                               random_seed=0, shuffle=False, vae_path=None, 
                                                epoch_freeze=False, argmax=True
                                               )