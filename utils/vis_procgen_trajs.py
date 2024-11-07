#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:17:02 2023

@author: x4nno
"""


import os
# sys.path.append("/home/x4nno/Documents/PhD/FRACOs_vg")
import sys
sys.path.append('.')
sys.path.append('..')

import time
import numpy as np
import pickle
# from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import copy

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
# from cuml.cluster import hdbscan
import hdbscan
import umap.umap_ as umap
import random
# from umap.parametric_umap import ParametricUMAP

from utils.eval import create_trajectories, create_random_trajectories, create_ind_opt_trajectories, create_even_trajectories
from cycler import cycler
from utils.VAE_creation import transform_obs_0_vae
from utils.VAE_CNN_creation import get_model
from utils.VAE_creation_fracos import VAE, VAE_procgen
import torch
from torch import nn

import gym as gym_old
import procgen

import gymnasium as gym
import pickle
from utils.compression import cluster_PI_compression
from utils.VAE_CNN_creation import get_model

from collections import Counter
from utils.sync_vector_env import SyncVectorEnv

import MetaGridEnv
from gym.envs.registration import register 

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from utils.compatibility import EnvCompatibility

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
    
from dataclasses import dataclass
import tyro

from gym import Wrapper

directions_dict = {
    0: ("LEFT", "DOWN"),
    1: ("LEFT",),
    2: ("LEFT", "UP"),
    3: ("DOWN",),
    4: (),
    5: ("UP",),
    6: ("RIGHT", "DOWN"),
    7: ("RIGHT",),
    8: ("RIGHT", "UP"),
    9: ("D",),
    10: ("A",),
    11: ("W",),
    12: ("S",),
    13: ("Q",),
    14: ("E",)
}

colors = ["#EBAC23",
            "#B80058",
            "#008CF9",
            "#006E00",
            "#00BBAD",
            "#D163E6",
            "#B24502",
            "#FF9287",
            "#5954D6",
            "#00C6F8",
            "#878500",
            "#00A76C",
            "#F6DA9C",
            "#FF5CAA",
            "#8ACCFF",
            "#4BFF4B",
            "#6EFFF4",
            "#EDC1F5",
            "#FEAE7C",
            "#FFC8C3",
            "#BDBBEF",
            "#BDF2FF",
            "#FFFC43",
            "#65FFC8",
            "#AAAAAA"]

def vis_trajs(trajectories, rewards):
    count = 0
    for t in trajectories:
        print("reward for this traj is ", rewards[count])
        placewait = input("hit enter to continue")
        plt.imshow([[0,0],[0,0]])
        plt.show()
        for o, a in t:
            plt.imshow(o/255, cmap="gray")
            try:
                a_num = np.where(a == 1)[0][0]
            except:
                a_num = a[0]
            a_name = directions_dict[a_num]
            plt.title(a_name)
            plt.show()
        count += 1

@dataclass
class Args:
    
    # Algorithm specific arguments
    env_id: str = "procgen-coinrun" 
    """the id of the environment MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun,
    atari:BreakoutNoFrameskip-v4"""
    failure_min: float = 0.97
    max_depth: int = 3
    max_clusters_per_clusterer: int = 20
    min_cluster_size: int = 100
    vae_path: str = None
    gen_strength: float = 0.1 # used in the supp fractures but also in the 
    chain_length: int = 2
    NN_predict: bool = False # do we use NN searches or just HDBSCAN -- false makes more sense.
    NN_cluster_search: bool = True # does nothing
    traj_refac: bool = True
    incremental: bool = False ## !!!CHANGE THIS BEFORE RUNNING AGAIN TO FALSE. Doesn't refactor if true.
    current_depth: int = 0
    min_PI_score: float = 0.3
    rm_fail: bool = True ## removes the failure trajectories instead of managing with a PI score
    supp_amount: int = 0
    NN_epochs: int = 1000
    style: str = "grid"
    vae: bool = True # should be false for default and running
    """use this tag to decide if a vae is needed -- this should only be used with procgen and atari."""
    vae_latent_shape: int = 10
    domain_size: int = 14
    max_ep_length: int = 1000
    
    
    top_only: bool = False # just to load the agent in
    
    debug: bool = False
  
if __name__ == "__main__":
    
    args = tyro.cli(Args)
    
    ## ** NEW WAY
    env_name = args.env_id
    if args.style == "grid":
        saved_traj_dir = f"trajectories/e2e_traj/{args.env_id}/"
    else:
        saved_traj_dir = f"trajectories/e2e_traj/{args.env_id}/{args.style}/"
        
    traj_path = saved_traj_dir+"trajs.p"
    rew_path = saved_traj_dir+"rews.p"
    traj_content = pickle.load(open(traj_path, "rb"))
    rew_content = pickle.load(open(rew_path, "rb"))
    
    all_trajectories = traj_content
    all_ep_rewards = rew_content
    
    if args.rm_fail:
        success_idxs = [index for index, value in enumerate(all_ep_rewards) if value > args.failure_min]
        all_trajectories = [all_trajectories[idx] for idx in success_idxs]
        all_ep_rewards = [all_ep_rewards[idx] for idx in success_idxs]
    
    
    vis_trajs(all_trajectories, all_ep_rewards)
    