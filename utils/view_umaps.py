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
import numpy as np

os.getcwd()

from train.fracos_all import train_pipe
from method.fracos_PPO import fracos_PPO
from method.fracos_QL import fracos_QL
from utils.eval import create_trajectories, create_random_trajectories, create_ind_opt_trajectories
from utils.clustering import create_fractures, create_clusterer,\
get_all_s_f_index, visualize_clusters, produce_umap_reducer_pipeline, \
produce_umap_reducer_pipeline_multi
from utils.compression import cluster_PI_compression
from utils.clustering import create_clusterer, colour_clusters_on_UMAP

from matplotlib import pyplot as plt

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

max_ep_len = 200                    # max timesteps in one episode
max_training_timesteps = int(1e6)   # break training loop if timeteps > max_training_timesteps

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
CHAIN_LENGTH = 5
H_level_pretrained = CURRENT_DEPTH

max_clusters_per_clusterer=0   # This will limit the maximum clusters at each depth
gen_strength=0.1                # 1 minus how strongly the clusterer should be sure that 
                                # a new point belongs to a cluster. at 0 no 
                                # inference or clusters will be used.

update_timestep = max_ep_len*5      # update policy every n timesteps

K_epochs = 40               # update policy for K epochs
eps_clip = 0.4              # clip parameter for PPO
gamma = 0.99                # discount factor

# lr_actor = 0.0003       # learning rate for actor network
# lr_critic = 0.001       # learning rate for critic network

lr_actor = 0.00002       # learning rate for actor network
lr_critic = 0.0002       # learning rate for critic network

chain_length = CHAIN_LENGTH
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
    env.seed(777)
else:
    env = gym.make(env_name)
    
plt.imshow(env.env_master.domain)    
plt.show()

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
                                      vae_path=None#"/home/x4nno/Documents/PhD/FRACOs_v3/pretrained/VAE/MetaGridEnv/vae_cnn.torch"
                                      )

checkpoint_dir = "/home/x4nno/Documents/PhD/FRACOs_v4.1/pretrained/fracos_PPO_preTrained/MetaGridEnv_Four_rooms"
checkpoint_ends = os.listdir(checkpoint_dir)
checkpoints = [checkpoint_dir+"/" + ce for ce in checkpoint_ends]

checkpoints = ["/home/x4nno/Documents/PhD/FRACOs_v4.1/pretrained/fracos_PPO_preTrained/MetaGridEnv_Four_rooms/fracos_PPO_MetaGridEnv_Four_rooms_0_0_10001.pth"
            #, "/home/x4nno/Documents/PhD/FRACOs_v4.1/pretrained/fracos_PPO_preTrained/MetaGridEnv_Four_rooms/fracos_PPO_MetaGridEnv_Four_rooms_0_0_60006.pth"
                , 
                "/home/x4nno/Documents/PhD/FRACOs_v4.1/pretrained/fracos_PPO_preTrained/MetaGridEnv_Four_rooms/fracos_PPO_MetaGridEnv_Four_rooms_0_0_110011.pth"
              ]

labels = [10000,
          #50000,
          100000]
print(checkpoints)

umap_save_location = "umap_reducers/MetaGridEnv/embeddings"

# We can pass offline trajectories into this pipeline and everything should work as planned ..

offline_path = "trajectories/opt/{}/{}".format(env_name, CURRENT_DEPTH)

list_of_trajectories, list_of_rewards, list_of_fractures, embedding, reducer = \
    produce_umap_reducer_pipeline_multi(fracos_agent, checkpoints, env, umap_save_location,
                                    env_name, labels, max_timesteps=5000, evalu=False,
                                    chain_length=CHAIN_LENGTH, include_rand_agent=True, one_level=False,
                                    offline_path=offline_path, min_traj_len=10 # this will make this use offline data
                                    )

## Now we want to visualise the clusters! 
MIN_CLUSTER_SIZE = 15
clusterer = create_clusterer(embedding, MIN_CLUSTER_SIZE=MIN_CLUSTER_SIZE)
pickle.dump(clusterer, open("utils/vis_of_clusterer.pkl", "wb"))

colour_clusters_on_UMAP(embedding, clusterer, env_name, legend=True)

## Save all when happy, then we can move on to creating visuals of exactly the options etc.
## To get clear clusters in 2d you might have to change the n_neighbors tag.

# PI
from utils.compression import cluster_PI_compression
from utils.clustering import get_all_s_f_index, visualize_clusters, visualize_cluster_one

all_fractures = sum(list_of_fractures, [])
ep_rewards = sum(list_of_rewards, [])
all_trajectories = sum(list_of_trajectories, [])

failure_std_threshold = None
failure_min = 0.95

all_s_f = get_all_s_f_index(all_fractures, ep_rewards, failure_std_threshold,
                      use_std=False, failure_min=failure_min)
    
concat_fractures = sum(all_fractures, [])
concat_trajs = sum(all_trajectories, [])

min_PI_value = 0.1

clusterer, top_cluster, all_success_clusters,\
        ordered_cluster_pi_dict, best_clusters_list = \
                cluster_PI_compression(clusterer, concat_fractures, all_s_f, all_trajectories,
                                       chain_length=CHAIN_LENGTH, max_cluster_returns=100, 
                                       min_PI_score = min_PI_value)

print(best_clusters_list)

for cluster_to_vis in best_clusters_list:
    labels = clusterer.labels_
    indexes = np.where(labels == cluster_to_vis)[0]
    # print(indexes)
    all_traj_to_cluster = []
    true_traj_to_cluster = []
    for index in indexes:
        all_traj_to_cluster.append(concat_fractures[index])
    
    seen_fractures = []
    
    plt.imshow(np.array([[cluster_to_vis],[cluster_to_vis]]))
    plt.colorbar()
    plt.show()
    for sample in range(len(all_traj_to_cluster)):
        if all_traj_to_cluster[sample] not in seen_fractures:
            seen_fractures.append(all_traj_to_cluster[sample])
            visualize_cluster_one(all_traj_to_cluster, concat_trajs, clusterer, cluster_to_vis,
                                  fracos_agent, sample, average=False,
                                   env_name="MetaGridEnv", vae=False, MAX_DEPTH=MAX_DEPTH,
                                   chain_length=CHAIN_LENGTH)

