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
from imblearn.pipeline import Pipeline
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

from collections import Counter, defaultdict
from utils.sync_vector_env import SyncVectorEnv

from gym.envs.registration import register 

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.optim as optim
from sklearn.metrics import accuracy_score

from utils.compatibility import EnvCompatibility

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
    
from dataclasses import dataclass
import tyro

from imblearn.combine import SMOTEENN, SMOTETomek 
from imblearn.over_sampling import SMOTE

from gym import Wrapper

from utils.default_networks import DefaultInitClassifier, MultiClassClassifier

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

def vis_trajs(trajectories):
    for t in trajectories:
        placewait = input("hit enter to continue")
        plt.imshow([[0,0],[0,0]])
        plt.show()
        for o, a in t:
            plt.imshow(o/255, cmap="gray")
            a_num = np.where(a == 1)[0][0]
            a_name = directions_dict[a_num]
            plt.title(a_name)
            plt.show()
            
            
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlattenObservationWrapper, self).__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.flatten(),
            high=env.observation_space.high.flatten(),
            dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        return observation.flatten()


@dataclass
class Args:
    
    # Algorithm specific arguments
    env_id: str = "procgen-starpilot" 
    """the id of the environment MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun,
    atari:BreakoutNoFrameskip-v4, highway:highway-fast-v0"""
    failure_min: float = 7.5
    max_depth: int = 1
    max_clusters_per_clusterer: int = 20
    min_cluster_size: int = 5
    vae_path: str = None
    gen_strength: float = 0.1 # used in the supp fractures but also in the 
    chain_length: int = 3
    NN_predict: bool = False # do we use NN searches or just HDBSCAN -- false makes more sense.
    NN_cluster_search: bool = True # does nothing
    traj_refac: bool = True
    incremental: bool = False ## !!!CHANGE THIS BEFORE RUNNING AGAIN TO FALSE. Doesn't refactor if true.
    current_depth: int = 0
    min_PI_score: float = 0.1
    rm_fail: bool = False ## removes the failure trajectories instead of managing with a PI score
    supp_amount: int = 25
    NN_epochs: int = 20
    style: str = "grid"
    vae: bool = True # should be false for default use args in start script to change
    """use this tag to decide if a vae is needed -- this should only be used with procgen and atari."""
    vae_latent_shape: int = 20
    domain_size: int = 14
    max_ep_length: int = 1000
    
    
    top_only: bool = False # just to load the agent in
    resnet_features: bool = False
    
    debug: bool = False
    
    
def custom_class_weights(y):
    # Calculate class frequencies
    class_counts = np.bincount(y)
    
    # Calculate weights based on custom logic
    # For example, penalize false positives more than false negatives
    weights = 1/class_counts
    
    weights[-1] = 1  # weight for no_label class (negative class)
    weights[:-1] = 2  # weight for all other classes (positive classes)
    
    return weights


def show_all_clusters(agent):
    for clstrr in range(len(agent.clusterers)):
        for clustr in agent.clusters[clstrr]:
            plt.imshow([[0, 1, 0, 1],
                        [0, 1, 1, 1],
                        [0, 1, 0, 1]])
            plt.show()
            visualize_clusters_deep(clstrr+1, clustr, agent)
            

def label_indices(original_array, unique_labels):
    # Create a dictionary to store the indices of unique labels
    label_dict = {label: index for index, label in enumerate(unique_labels)}
    
    # Map the labels in the original array to their indices in the unique labels list
    indices = np.array([label_dict[label] for label in original_array])
    
    return indices


def make_env(env_id, idx, capture_video, run_name, args):
    def thunk():
        
        if "procgen" in args.env_id: # need to create a specific method for making these vecenvs
            env = gym_old.make(args.env_id, num_levels=100, start_level=1, distribution_mode="easy") # change this will only do one env
            env.observation_space = gym.spaces.Box(0,255,(64,64,3), "int")
            env.action_space = gym.spaces.Discrete(env.action_space.n)
            
            env.action_space
            #envs = gym_old.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
            
            env = EnvCompatibility(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym_old.wrappers.NormalizeReward(env, gamma=0.99)
            env = gym_old.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            
            ## Needs a wrapper to turn to from gym.spaces to gymnasium.spaces for both obs and action?
            
        elif "atari" in env_id:
            env_in_id = env_id.split(":")[-1] # because we only want the name but to distinguish we need the atari:breakout etc
            env = gym.make(env_in_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            # env = gym.wrappers.RecordEpisodeStatistics(env)
            # if capture_video:
            #     if idx == 0:
            #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
            
            
        elif "highway" in env_id:
            env_in_id = env_id.split(":")[-1]
            env = gym.make(env_in_id)
            env = FlattenObservationWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        
        
        elif "metagrid" in env_id:
            env = gym.make(env_id, domain_size = [args.domain_size, args.domain_size], max_episode_steps=args.max_ep_length)
        else:
            env = gym.make(env_id, max_episode_steps=args.max_ep_length)
        
        
        env = fracos_ppo_wrapper(env, 1e8)
        
        return env

    return thunk



class fracos_ppo_wrapper(Wrapper):
    def __init__(self, env, max_ep_length):
        super().__init__(env)
        self.max_ep_length = max_ep_length
        
    def fracos_step(self, action, next_ob, agent, total_rewards=0, total_steps_taken=0, agent_update=False, top_level=False):
        all_info = {}
        
        try:
            ob = tuple(next_ob.cpu().numpy())
        except:
            ob = tuple(next_ob)
        
        if action not in range(agent.action_prims):
            if agent.top_only and top_level:
                action = action - agent.total_action_dims + agent.all_unrestricted_actions
            
            if ob not in agent.discrete_search_cache.keys():
                agent.initial_search(ob)
            id_actions = tuple(agent.discrete_search_cache[ob][action])
            if isinstance(id_actions[0], np.ndarray):
                for id_action in id_actions:
                    
                    if "HRL issue" in all_info:
                        break # do not carry on if failed on this!
                    
                    for reverse_cypher in agent.reverse_cyphers:
                        if tuple(id_action) in reverse_cypher.keys():
                            id_action = reverse_cypher[tuple(id_action)]
                            break
                        
                    next_ob, total_rewards, reward, termination, truncation, info, total_steps_taken = \
                        self.fracos_step(id_action, next_ob, agent, total_rewards=total_rewards, total_steps_taken=total_steps_taken)
                        
                    all_info.update(info)
                    
                    # need to exit if we have finished
                    next_done = np.logical_or(termination, truncation)
                    if next_done:
                        return next_ob, total_rewards, reward, termination, truncation, info, total_steps_taken
                        
            else:
                # returns a negative reward and our current location.
                
                #This is handled through the INFO tag to stop it happening again,
                return ob, 0, 0, False, None, {"HRL issue" : True}, total_steps_taken
        else:
            next_ob, reward, termination, truncation, info = self.env.step(action)
            total_rewards += reward
            total_steps_taken += 1
            next_done = np.logical_or(termination, truncation)
            if next_done:
                return next_ob, total_rewards, reward, termination, truncation, info, total_steps_taken
            
        if self.env.episode_lengths > self.max_ep_length:
            truncation = True
        else:
            truncation = False
            
        dones = np.logical_or(termination, truncation)
                
        num_dones = np.sum(dones)
        if num_dones:
            if "episode" in info or "_episode" in info:
                raise ValueError(
                    "Attempted to add episode stats when they already exist"
                )
            else:
                info["episode"] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(
                        dones,
                        np.round(time.perf_counter() - self.episode_start_times, 6),
                        0.0,
                    ),
                } 
        
        # if I unhash the below, we end up with HUGE KL divergence
        # This is because we change the logprobs from near 0 to -1e-6.
        # It doesn't damage the running. but you can't limit updates on KL anymore.
        if "HRL issue" in info:
            with torch.no_grad():
                agent.discrete_search_cache[ob][action] = [None, None]
        
        return next_ob, total_rewards, reward, termination, truncation, info, total_steps_taken

def remove_everything_in_folder(folder_path):
    """
    Remove all files and subdirectories in a folder.

    Args:
    - folder_path (str): Path of the folder.

    Returns:
    - None
    """
    # List all files and directories in the folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Remove all files
        for file in files:
            file_path = os.path.join(root, file)
            if "a/" not in file_path:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        # Remove all subdirectories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if "/a" not in dir_path:
                os.rmdir(dir_path)
                print(f"Removed directory: {dir_path}")
    
    print(f"All contents in folder '{folder_path}' have been removed. except policy.pth")


def show_from_state(state):
    plt.imshow(state[:49].reshape(7,7))
    
def vis_all_envs(env_name):
    env_dir = "trajectories/opt" + f"/{env_name}/0/envs/"
    for f in os.listdir(env_dir):
        fd = os.path.join(env_dir, f)
        with open(fd, "rb") as file:
            env_list = pickle.load(file)
            for env in env_list:
                plt.imshow(env.env_master.domain)
                plt.show()
                
                
def get_encodings(obs, vae_path, model):
    with torch.no_grad():
        input_height = obs.shape[-1]
        input_width = obs.shape[-1]
        
        h = model.encoder(obs)
        h = h.reshape(h.size(0), -1)
        mu = model.fc_mu(h)
        logvar = model.fc_logvar(h)
        comp_obs = model.reparameterize(mu, logvar)
        comp_obs = comp_obs.cpu().detach().numpy()
        
        return comp_obs
    
    

def create_fractures(trajectories, env_name, obs_space=False, vae_path="pretrained/VAE/MetaGridEnv/vae_cnn.torch",
                     chain_length=2, a_pre_enc=True, fracos_agent=None):
    
    if (not args.vae) and ("MetaGridEnv" in env_name):

        all_fractures = []
        corre_traj = []
        for trajectory in trajectories:
            try:
                move_count=0
                for move in trajectory:
                    state_count = 0
                    for state in move:
                        if torch.is_tensor(state):
                            trajectory[move_count][state_count] = state.cpu().detach()
                        state_count += 1
                    move_count += 1
                    
                trajectory = np.array(trajectory, dtype=object)
                states = trajectory[:,0]
                obs_0_list = []
                obs_0_noenc_list = []
                obs_1_list = []
                for state in states:
                    obs_0 = state[:49]
                    obs_1 = state[49:]
                    obs_0_list.append(obs_0)
                    obs_1_list.append(obs_1)
                
                obs_0_arr = np.vstack(obs_0_list)
                obs_1_arr = np.vstack(obs_1_list)
                
                obs = np.concatenate((obs_0_arr, obs_1_arr), axis=1)
                
                obs = obs[:-(chain_length)]
                
                
                actions1 = trajectory[:-(chain_length),1]
                if a_pre_enc:
                    actions1 = np.array([fracos_agent.cypher[int(item[0])] for item in actions1])
                    # actions1 = fracos_agent.cypher[int(actions1)]
                actions1 = np.asarray(actions1)
                actions1 = np.stack(actions1)
                
                frac = np.concatenate((obs, actions1), axis=1)
                
                for b in range(1,chain_length):
                    n_actions = trajectory[b:-(chain_length-b),1]
                        
                    if a_pre_enc:
                        n_actions = np.array([fracos_agent.cypher[int(item[0])] for item in n_actions])
                    n_actions = np.asarray(n_actions)
                    n_actions = np.stack(n_actions)
                    frac =  np.concatenate((frac, n_actions), axis=1)
                
                frac = frac.tolist()
                
                all_fractures.append(frac)
            except:
                print("A trajectory has been found which is shorter than chain length, please reduce chain length")
        
    elif (not args.vae):
        
        all_fractures = []
        corre_traj = []
        for trajectory in trajectories:
            move_count=0
            for move in trajectory:
                state_count = 0
                for state in move:
                    if torch.is_tensor(state):
                        trajectory[move_count][state_count] = state.cpu().detach()
                    state_count += 1
                move_count += 1
                
            trajectory = np.array(trajectory, dtype=object)
            states = trajectory[:,0]
            state_list = []
            for state in states:
                state_list.append(state)
                
            state_list_arr = np.vstack(state_list)
            
            obs = state_list_arr[:-(chain_length)]
            
            
            actions1 = trajectory[:-(chain_length),1]
            if a_pre_enc:
                actions1 = np.array([fracos_agent.cypher[int(item[0])] for item in actions1])
            actions1 = np.asarray(actions1)
            actions1 = np.stack(actions1)
            
            frac = np.concatenate((obs, actions1), axis=1)
            
            for b in range(1,chain_length):
                n_actions = trajectory[b:-(chain_length-b),1]
                    
                if a_pre_enc:
                    n_actions = np.array([fracos_agent.cypher[int(item[0])] for item in n_actions])
                n_actions = np.asarray(n_actions)
                n_actions = np.stack(n_actions)
                frac =  np.concatenate((frac, n_actions), axis=1)
            
            frac = frac.tolist()
            
            all_fractures.append(frac)
            
    elif (args.vae) and (("procgen" in args.env_id) or ("atari" in args.env_id)):
        if not args.vae_path:
            vae_path = f"vae_models/{args.env_id}/model.pth"
        else:
            vae_path = args.vae_path
            
        if "procgen" in args.env_id:
            model = VAE_procgen(3, args.vae_latent_shape, 64, 64).to(device)
            model.load_state_dict(torch.load(vae_path))
            model.eval()
        
        
        
        all_fractures = []
        corre_traj = []
        for trajectory in trajectories:
            if len(trajectory) < 10:
                pass # this is because in some environments procgen has bugs where it spawns on something?
            else:
                move_count=0
                for move in trajectory:
                    state_count = 0
                    for state in move: 
                        if torch.is_tensor(state):
                            trajectory[move_count][state_count] = state.cpu().detach()
                        state_count += 1
                    move_count += 1
                    
                trajectory = np.array(trajectory, dtype=object)
                # print("len of trajectory", len(trajectory))
                states = trajectory[:,0]
                state_list = []
                for state in states:
                    
                    state_list.append(state)
                    
                state_list_arr = np.stack(state_list)
                
                obs = state_list_arr[:-(chain_length)]
                
                # here is where the vae needs to go!
                
                if "procgen" in args.env_id:
                    # process the obs 
                    obs = np.transpose(obs, (0, 3, 1, 2 ))
                    obs.astype(np.float32) / 255.0
                    obs = torch.from_numpy(obs).to(device)
                    # get_encodings 
                    print("length of obs ", len(obs))
                    comp_obs = get_encodings(obs, vae_path, model)
                    
                
                actions1 = trajectory[:-(chain_length),1]
                
                # print("actions1 are : ")
                # print(actions1)
                
                if a_pre_enc:
                    actions1 = np.array([fracos_agent.cypher[int(item[0])] for item in actions1])
                actions1 = np.asarray(actions1)
                actions1 = np.stack(actions1)
                
                frac = np.concatenate((comp_obs, actions1), axis=1)
                
                for b in range(1,chain_length):
                    n_actions = trajectory[b:-(chain_length-b),1]
                        
                    if a_pre_enc:
                        n_actions = np.array([fracos_agent.cypher[int(item[0])] for item in n_actions])
                    n_actions = np.asarray(n_actions)
                    n_actions = np.stack(n_actions)
                    frac =  np.concatenate((frac, n_actions), axis=1)
                
                frac = frac.tolist()
                
                all_fractures.append(frac)
                
    # all_fractures is now a list of all fractures in all the trajectories
            
    return all_fractures, corre_traj


def get_all_s_f_index(all_fractures, ep_rewards, failure_std_threshold,
                      use_std=True, failure_min=None):
    
    """Returns the a list of 1 for success and 0 for failure for every fracture """
    
    rew_mean = np.mean(ep_rewards) 
    rew_std = np.std(ep_rewards)
    
    if use_std:
        failure_threshold = rew_mean - failure_std_threshold*rew_std
    else:
        failure_threshold = failure_min
    
    failure_indexes = np.where(np.asarray(ep_rewards) < failure_threshold)[0]
    
    all_s_f = []
    for i in range(len(all_fractures)):
        for j in all_fractures[i]:
            if i in failure_indexes:
                all_s_f.append(0)
            else:
                all_s_f.append(1)
                
    return all_s_f


    
def create_clusterer(all_fractures, MIN_CLUSTER_SIZE=30, metric="euclidean", simple=True):
    # WE MAY want to use umap as the reduction technique to cluster!?
    # instead of a VAE?
    
    if simple:
        pass
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, gen_min_span_tree=True,
                                prediction_data=True, metric=metric)
    
    if not isinstance(all_fractures, np.ndarray): # as will be a list of lists
        concat_fractures = sum(all_fractures, [])
        concat_fractures = np.asarray(concat_fractures)
    
    else:
        concat_fractures = all_fractures
    
    clusterer.fit(concat_fractures)
    
    return clusterer



def find_goal_from_dirdis(direction, distance):
    # print(direction, distance)
    
    if 0 < direction <= np.pi/2: # top right
        x_ref = distance*np.cos(direction)
        y_ref = -distance*np.sin(direction)
    elif np.pi/2 < direction <= np.pi: # top left
        x_ref = -distance*np.cos(np.pi - direction)
        y_ref = -distance*np.sin(np.pi-direction)
    elif -np.pi < direction <= -np.pi/2: # bottom left
        x_ref = -distance*np.cos(np.pi+direction)
        y_ref = distance*np.sin(np.pi+direction)
    elif -np.pi/2 < direction <= 0: # bottom right
        x_ref = distance*np.cos(-direction)
        y_ref = distance*np.sin(-direction)
        
    return x_ref, y_ref


            
def act_no_to_clstr(act_no, fracos_agent):
    "Only for MetaGridEnv"
    count = 4 # start 4 for primitive
    clstrs_count = -1
    if act_no < 4:
        return clstrs_count, act_no
    clstrs_count = 0
    for clstrs in fracos_agent.clusters:
        if act_no < count + len(clstrs):
            return clstrs_count, clstrs[act_no-count]
        else:
            count = count + len(clstrs)
        clstrs_count += 1
        
def visualize_clusters_deep(depth, cluster_to_vis, method,
                            average=False, samples=1, env_name="MetaGridEnv/metagrid-v0",
                            vae=False, MAX_DEPTH=3, chain_length=2, from_offline_prim=True,
                            ):
    
    if "MetaGridEnv" not in env_name:
        print("""This function currently only works for MetaGridEnv - to use it for 
              other environments you will need to change this function""")
        return    
    
    cluster_dir="fracos_clusters/"
    
    clusterer = pickle.load(open(cluster_dir + env_name + "/clusterers/" + "clusterer{}.p".format(depth-1), "rb"))
    concat_fractures = pickle.load(open(cluster_dir + env_name + "/other/" + "concat_fractures{}.p".format(depth-1), "rb"))
    concat_trajs = pickle.load(open(cluster_dir + env_name + "/other/" + "concat_trajs{}.p".format(depth-1), "rb"))
    # cyphers
    cypher = pickle.load(open(cluster_dir + env_name + "/cluster_cyphers/"+ "cypher_{}.p".format(depth-1), "rb"))
    reverse_cypher = pickle.load(open(cluster_dir + env_name +"/cluster_reverse_cyphers/" + "cypher_{}.p".format(depth-1), "rb"))
    
    labels = clusterer.labels_
    indexes = np.where(labels == cluster_to_vis)[0]
    # print(indexes)
    all_traj_to_cluster = []
    true_traj_to_cluster = []
    for index in indexes:
        all_traj_to_cluster.append(concat_fractures[index])
        if vae:
            true_traj_to_cluster.append(concat_trajs[index])
    
    # Either average or produce several visualisations
    if average:
        print("not implemented yet")
    else:
        if vae:
            figure, axis = plt.subplots(2, samples)
        else:
            figure, axis = plt.subplots(1, samples) # change back to 2 if having issues
        for i in range(samples):
            random_idx = random.choice(range(len(all_traj_to_cluster)))
            random_traj = all_traj_to_cluster[random_idx]
            if vae:
                random_true_traj = true_traj_to_cluster[random_idx]
            # print(random_traj)
            if not from_offline_prim:
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    action_list = []
                    for a in range(chain_length):
                        # print(6+a+a*MAX_DEPTH)
                        # print(7+a+(a+1)*MAX_DEPTH)
                        # print("------")
                        acti = random_traj[6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim): \
                                           6+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim)]
                        # print(6+a*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        # print(6+a+(a+1)*(MAX_DEPTH*method.max_clusters_per_clusterer+method.action_dim))
                        action_list.append(reverse_cypher[tuple(acti)])
                else:
                    #print("NOT IMPLEMENTED CORRECTLY YET")
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) / chain_length
                    
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                
            else:
                # Will only work for one level. Will need to change for two levels.
                if vae:
                    state = method.decoder(torch.tensor(random_traj[:4]).view(1,4).to(device))
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    state = state.cpu().detach().numpy()
                    goal_location = random_traj[4:6]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) / chain_length
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                else:
                    state = np.asarray(random_traj[:49])
                    state = state.reshape((7,7))
                    state[3,3] = 2 # agent_loc
                    goal_location = random_traj[49:51]
                    all_actions = random_traj[51:]
                    even_splits = len(all_actions) // chain_length 
                    pre_action_list = [all_actions[i*even_splits: (i+1) * even_splits] for i in range(chain_length)]
                    action_list = [reverse_cypher[tuple(ac)] for ac in pre_action_list]
                    
            x_ref, y_ref = find_goal_from_dirdis(goal_location[0], goal_location[1])
            x_ref = int(x_ref)+3
            y_ref = int(y_ref)+3
            # print(state)
            # if (x_ref <= 3) and (y_ref <= 3): # keep within image
            #     state[x_ref, y_ref] = 3 # goal loc
            
            goal_location = [ '%.2f' % elem for elem in goal_location]
            
            
            if vae:
                axis[0, i].imshow(state, cmap='Greys')
                axis[0, i].set_title("{}, {}".format(goal_location, action_list), fontsize=6)
                
                true_state = np.asarray(random_true_traj[0][:49]).reshape((7,7))
                true_state[3,3] = 2 # agent_loc
                true_goal_location = random_true_traj[0][49:51]
                true_goal_location = [ '%.2f' % elem for elem in true_goal_location]
                
                axis[1, i].imshow(true_state, cmap='Greys')
                axis[1, i].set_title("{}, {}".format(true_goal_location, action_list), fontsize=6)
                # axis[0, i].set_xticks([])
                # axis[0, i].set_yticks([])   
                    
            elif samples == 1:
                axis.imshow(state, cmap="Greys") 
                axis.set_title("{}, {}".format(goal_location, action_list), fontsize=6)
            else:
                axis[i].imshow(state, cmap='Greys')
                axis[i].set_title("{}, {}".format(goal_location, action_list), fontsize=6)

        plt.setp(axis, xticks=[], yticks=[])
        figure.suptitle("{}: Cluster {}".format(depth, cluster_to_vis))
        #plt.colorbar()
        plt.show()
        plt.clf()
        
        for a in action_list:
            if a > 3: # then this is not a primitive and we should print out what it is.
                clstr_idx, cluster = act_no_to_clstr(a, method)
                visualize_clusters_deep(clstr_idx+1, cluster, method, env_name=env_name,
                                        average=average, samples=samples,
                                        vae=vae, chain_length=chain_length,
                                        from_offline_prim=from_offline_prim)


class torch_classifier(nn.Module):
    def __init__(self, fracture_shape, num_labels, dropout_prob=0.4):
        super().__init__()
        initial_shape = fracture_shape
        self.layer_1 = nn.Linear(in_features=initial_shape, out_features=256)
        self.layer_2 = nn.Linear(in_features=256, out_features=256)
        self.layer_3 = nn.Linear(in_features=256, out_features=256)
        self.layer_4 = nn.Linear(in_features=256, out_features=num_labels)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
    # 3. Define a forward method containing the forward pass computation
    def forward(self, x):
        x = self.leaky_relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.layer_4(x))
        return x

def accuracy_fn(y_true, y_pred):
    try:
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    except:
        masked_tensor = y_true.where(y_true != -1, torch.tensor(float('-inf'), device=y_true.device))

        # Find the maximum ignoring -1
        max_labels, _ = torch.max(masked_tensor, dim=1)
        correct = torch.eq(max_labels, y_pred).sum().item()
        
    acc = (correct / len(y_pred)) * 100
    return acc

def supplement_fractures(fractures, labels, clusterer, all_possible_action_combs, env,
                         gen_str = 0.2, supp_amount=50, vae_path=None, no_label=666):
    
    check_env = True
    supp_fractures = []
    supp_labels = []
    for idx in range(len(fractures)):
        if idx%1000 == 0:
            print(f"{idx} / {len(fractures)}")
        if (labels[idx] != no_label) or (labels[idx]  != -1):
            if args.vae:
                # 6 is the latent VAE shape + dir dis. Needs to be changed depending on VAE
                frac_0 = np.array(fractures[idx][:args.vae_latent_shape])
            else:
                try:
                    frac_0 = fractures[idx][:env.observation_space.shape[0]]
                except:
                    frac_0 = np.array([fractures[idx][0],])
                    

                # i = 0
                # while i < supp_amount:
                #     apac = random.choice(all_possible_action_combs)
                #     first = True
                #     for a in apac:
                #         if first:
                #             ac = a
                #             first = False
                #         else:
                #             ac = np.concatenate((ac, a))
                #     new_frac = np.concatenate((np.asarray(frac_0),ac))
                #     if list(new_frac) not in fractures:
                #         supp_fractures.append(new_frac)
                #         supp_labels.append(no_label)
                #         i += 1
                
            i = 0
            fractures_set = set(map(tuple, fractures))  # Convert fractures to a set of tuples for faster lookup
            while i < supp_amount:
                apac = random.choice(all_possible_action_combs)
                ac = np.concatenate(apac)  # Direct concatenation of all elements in apac
                new_frac = np.concatenate((frac_0, ac))
            
                if tuple(new_frac) not in fractures_set:  # Check against the set
                    supp_fractures.append(new_frac)
                    supp_labels.append(no_label)
                    fractures_set.add(tuple(new_frac))  # Keep the set updated
                i += 1
            
            if check_env == True:
                try:
                    tocheckforMG = env.env_master.domain # this is a MG specific supp for dirdis
                    # we do nothing with above but this is just to check it will fail if not mg
                    frac_1 =  fractures[idx][env.observation_space.shape[0]:]
                    frac_1 = np.array(fractures[idx][env.observation_space.shape[0]:])
                    for n in range(supp_amount):
                        new_frac_0 = copy.copy(frac_0)
                        new_frac_0[-1] = frac_0[-1]*(-1+random.uniform(-0.5,0.5))
                        new_frac_0[-2] = frac_0[-2]*(-1+random.uniform(-0.5,0.5))
                        new_frac = np.concatenate((np.asarray(new_frac_0),frac_1))
                        
                        if tuple(new_frac) not in fractures_set:  # Check against the set
                            supp_fractures.append(np.array(new_frac))
                            supp_labels.append(no_label)
                            fractures_set.add(tuple(new_frac))  # Keep the set updated
                            
                        
                        new_frac_0 = copy.copy(frac_0)
                        new_frac_0[-1] = frac_0[-1]*(1+random.uniform(-0.5,0.5))
                        new_frac_0[-2] = frac_0[-2]*(1+random.uniform(-0.5,0.5))
                        new_frac = np.concatenate((np.asarray(new_frac_0),frac_1))
                        
                        if tuple(new_frac) not in fractures_set:  # Check against the set
                            supp_fractures.append(np.array(new_frac))
                            supp_labels.append(labels[idx])
                            fractures_set.add(tuple(new_frac))  # Keep the set updated
                    
                    
                except:
                    check_env = False
                    # print("not MetaGridEnv")
                    pass
                
                    
        # print("{}/{}".format(idx, len(fractures)))
    
    supp_fractures = np.asarray(supp_fractures)
    
    # _, strengths = hdbscan.approximate_predict(clusterer, fractures)
    
    all_fractures = np.concatenate((fractures, supp_fractures))
    
    labels = np.concatenate((labels, supp_labels))
    
    # unhash to remove any cluster predictions which don't pass the gen_str barrier.
    
    # for i in range(len(strengths)):
    #     if strengths[i] <= 1-gen_str:
    #         labels[i] = no_label       
        

    
    # print("Oversampling for labels")
    # all_fractures, labels = ros.fit_resample(all_fractures, labels)
    
            
    return all_fractures, labels
        
def find_single_occurrence_numbers_with_indices(lst):
    counts = Counter(lst)
    single_occurrence_data = [(num, index) for index, num in enumerate(lst) if counts[num] == 1]
    return single_occurrence_data


def remove_single_occurrence_numbers(lst):
    sod = find_single_occurrence_numbers_with_indices(lst)
    for s in sod:
        lst[s[1]] = -1
    
    return lst

def create_NN_clusters(fractures, labels, clusterer, all_possible_action_combs, env,
                       verbose=False, max_epochs=30000, gen_str=0.2, chain_len=2,
                       vae_path=None):
    
    # supplement fractures with 0 labels
    old_fractures = copy.deepcopy(fractures)
    old_labels = copy.deepcopy(clusterer.labels_)
    
    eval_X = torch.tensor(old_fractures).to(device)
    eval_y = torch.tensor(old_labels).type(torch.LongTensor).to(device)
    
    class_weights = None
    
    print("Supplementing Fractures")
    
    # only care about those which are labelled the same as our clusters, we will increase to one above 
    # new_labels = []
    # no_label = max(clusterer.labels_) + 1
    # for l in clusterer.labels_:
    #     if (l not in labels) or (l==-1):
    #         new_labels.append(no_label)
    #     else:
    #         new_labels.append(l)
    
    no_label = max(clusterer.labels_) + 1
    new_labels = clusterer.labels_
    
    if args.supp_amount > 0:
        fractures, new_labels = supplement_fractures(fractures, new_labels, clusterer, 
                                                  all_possible_action_combs, env, supp_amount=args.supp_amount,
                                                  gen_str=gen_str, vae_path=vae_path, no_label=no_label)
    
    
    fractures = np.asarray(fractures)
    fractures = fractures.astype(float)
    
    # We need to remove any cluster which has only one label?
    unique_labels = set(new_labels)
    num_labels = len(unique_labels)
    
    # labels = remove_single_occurrence_numbers(labels)
    new_labels = [no_label if x == -1 else x for x in new_labels]
    labels.remove(-1)
    labels.append(no_label) # will always be at the end now.
    cat_labels = label_indices(new_labels, labels)
    
    # X_train, y_train = fractures, cat_labels
    
    # this is needed to rearrange the labels if using BCE loss or multi hinge 
    # NEED TO HASH THIS OUT IF NOT USING BCE style
    # num_samples = len(cat_labels)
    # num_classes = len(set(cat_labels))
    # cat_labels_t = torch.full((num_samples, num_classes), -1, dtype=torch.long)
    # for i, label in enumerate(cat_labels):
    #     cat_labels_t[i, 0] = label 
    # cat_labels = cat_labels_t
    
    X_train, X_test, y_train, y_test = train_test_split(fractures, 
                                                        cat_labels, 
                                                        stratify=cat_labels,
                                                        random_state=1)

    # Need to remove the 0 if not using BCE loss
    meaningful_idx = [index for index, element in enumerate(list(cat_labels)) if element != no_label]
    meaningful_X = fractures[meaningful_idx]
    meaningful_y = cat_labels[meaningful_idx]
    
    print("getting weights")
    # can add this back in if going to use weights.
    
    class_weights = custom_class_weights(y_train)

    # Create a dictionary mapping class indices to their corresponding weights
    class_weight_dict = dict(enumerate(class_weights))
    
    class_weights=compute_class_weight(class_weight_dict,classes=np.unique(y_train),y=y_train)
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    
    model_args = [len(fractures[0]), len(set(new_labels))] # could be cat_labels
    model_0 = torch_classifier(len(fractures[0]), len(set(new_labels))).to(device)
    
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        # loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
        # loss_fn = nn.BCEWithLogitsLoss()
        
    
    
    
        
    optimizer = torch.optim.Adam(params=model_0.parameters(), 
                            lr=3e-3)
    # optimizer = torch.optim.Adadelta(params=model_0.parameters(), rho=0.9)
    
    # optimizer = torch.optim.SGD(model_0.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    
    # Set the number of epochs
    
    # Put data to target device
    X_train, y_train = torch.tensor(X_train).to(device), torch.tensor(y_train).type(torch.LongTensor).to(device)
    X_test, y_test = torch.tensor(X_test).to(device), torch.tensor(y_test).type(torch.LongTensor).to(device)
    meaningful_X, meaningful_y = torch.tensor(meaningful_X).to(device), torch.tensor(meaningful_y).type(torch.LongTensor).to(device)
    
    
    
    X_train = X_train.float()
    X_test = X_test.float()
    meaningful_X = meaningful_X.float()
    
    # Build training and evaluation loop
    
    last_test_loss = 100000
    for epoch in range(max_epochs):
        # 1. Forward pass
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1) # logits -> prediction probabilities -> prediction labels
        
        # 2. Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        # print("y_train")
        # print(y_train)
        # print("y_pred")
        # print(y_pred)
        acc = accuracy_fn(y_true=y_train, 
                          y_pred=y_pred)
        
        # 3. Optimizer zero grad
        optimizer.zero_grad()
    
        # 4. Loss backward
        loss.backward()
    
        # 5. Optimizer step
        optimizer.step()
    
        ### Testing
        model_0.eval()
        with torch.inference_mode():
          # 1. Forward pass
          test_logits = model_0(X_test).squeeze()
          test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
          # 2. Calcuate loss and accuracy
          test_loss = loss_fn(test_logits, y_test)
          test_acc = accuracy_fn(y_true=y_test,
                                  y_pred=test_pred)
          
          mful_logits = model_0(meaningful_X).squeeze()
          mful_pred = torch.softmax(mful_logits, dim=1).argmax(dim=1)
          # 2. Calcuate loss and accuracy
          mful_loss = loss_fn(mful_logits, meaningful_y)
          mful_acc = accuracy_fn(y_true=meaningful_y,
                                  y_pred=mful_pred)
    
        # Print out what's happening
        if epoch % 100 == 0:
            if verbose:
                print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.2f}% | mful Loss: {mful_loss:.5f}, mful Accuracy: {mful_acc:.2f}%")
            
    
        # print("Final accuracy on all clusters (ignoring the outliers)")
        # with (torch.inference_mode()) and (torch.no_grad()):
        #     # 1. Forward pass
        #     final_logits = model_0(eval_X).squeeze()
        #     final_pred = torch.softmax(final_logits, dim=1).argmax(dim=1)
        #     # 2. Calcuate loss and accuracy
        #     final_loss = loss_fn(final_logits, eval_y)
        #     final_acc = accuracy_fn(y_true=eval_y,
        #                            y_pred=final_pred)
            
        #     print("Final loss = {}. Final accuracy = {}%".format(final_loss, final_acc))
    
    
    return model_0, model_args

def create_NN_clusters_test(train_fractures, test_fractures, train_labels, test_labels, verbose=False):
    cl_model = create_NN_clusters(train_fractures, train_labels, verbose=verbose)
    return cl_model

def save_all_clusterings(clusterer, clusters, concat_fractures, concat_trajs, NN, model_args, method, cluster_level, env_name):
    # check if dir exist if not make it.
    if not os.path.exists("fracos_clusters/" + env_name + "/clusterers"):
        os.makedirs("fracos_clusters/" + env_name + "/clusterers")
    if not os.path.exists("fracos_clusters/" + env_name + "/clusters"):
        os.makedirs("fracos_clusters/" + env_name + "/clusters")
    if not os.path.exists("fracos_clusters/" + env_name + "/other"):
        os.makedirs("fracos_clusters/" + env_name + "/other")
    if not os.path.exists("fracos_clusters/" + env_name + "/NNs"):
        os.makedirs("fracos_clusters/" + env_name + "/NNs")
    if not os.path.exists("fracos_clusters/" + env_name + "/NN_args"):
        os.makedirs("fracos_clusters/" + env_name + "/NN_args")
    if not os.path.exists("fracos_clusters/" + env_name + "/cluster_cyphers"):
        os.makedirs("fracos_clusters/" + env_name + "/cluster_cyphers")
    if not os.path.exists("fracos_clusters/" + env_name + "/cluster_reverse_cyphers"):
        os.makedirs("fracos_clusters/" + env_name + "/cluster_reverse_cyphers")
    
    
    pickle.dump(clusterer, open("fracos_clusters/" + env_name + "/clusterers/" + "clusterer{}.p".format(cluster_level), "wb"))
    pickle.dump(clusters, open("fracos_clusters/" + env_name + "/clusters/" + "clusters{}.p".format(cluster_level), "wb"))
    pickle.dump(concat_fractures, open("fracos_clusters/" + env_name + "/other/" + "concat_fractures{}.p".format(cluster_level), "wb"))    
    pickle.dump(concat_trajs, open("fracos_clusters/" + env_name + "/other/" + "concat_trajs{}.p".format(cluster_level), "wb")) 
    if NN is not None:
        torch.save(NN.state_dict(), "fracos_clusters/"+ env_name + "/NNs/" + "NN_state_dict_{}.pth".format(cluster_level))
    pickle.dump(model_args, open("fracos_clusters/" + env_name + "/NN_args/" + "NN_args_{}.p".format(cluster_level), "wb"))

    if method is not None:
        pickle.dump(method.cypher, open("fracos_clusters/" + env_name + "/cluster_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))
        pickle.dump(method.reverse_cypher, open("fracos_clusters/" + env_name + "/cluster_reverse_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))


def refactor_trajectories(trajectories, fracos_agent, env_name, env, depth, gen_strength,
                          max_clusters_per_clusterer, device="cuda", chain_length=2, NN_predict=False):
    
    # This should use the exact clusters and not the NN predictions? 
    
    # stop compliling errors?
    
    refac_trajs = []
    traj_count = 0
    for traj in trajectories:
        if (traj_count % 2) == 0: 
            print("starting {} of {} ".format(traj_count, len(trajectories)))
        traj_count += 1
        refac_traj = []
        option_found = False
        count = 0
        
        if args.vae and "procgen" in args.env_id:
            vae_path = f"vae_models/{args.env_id}/model.pth"
            model = VAE_procgen(3, args.vae_latent_shape, 64, 64).to(device)
            model.load_state_dict(torch.load(vae_path))
            model.eval()
        
        for move_idx in range(len(traj)-(chain_length-1)):
            move = traj[move_idx]
            if option_found and count < chain_length:
                # skip the next state and the next immediate analysis 
                option_found = False
                count += 1
            else:
                if "MetaGridEnv" in env_name:
                    state = move[0]
                    obs_0 = state[:49]
                    obs_0 = torch.tensor(obs_0).to(device)
                    obs_1 = state[49:]
                    obs_1 = torch.tensor(obs_1).to(device)
                    if fracos_agent.vae_path:
                        obs_0_enc = fracos_agent.encoder(obs_0.float().reshape(1,1,7,7))[0][0]
                    else:
                        obs_0_enc = obs_0
                        
                    obs_0_enc = obs_0_enc.detach().to("cpu")
                    obs_1 = obs_1.detach().to("cpu")
                    
                    state_c = np.concatenate((obs_0_enc, obs_1))
                    
                elif "procgen" in args.env_id:                    
                    state_c = move[0]
                    if state_c.shape[-1] == 3:
                        state_c = np.transpose(state_c, (2, 0, 1 ))
                    state_c = np.expand_dims(state_c, axis=0)
                    state_c.astype(np.float32) / 255.0
                    state_c = torch.from_numpy(state_c).to(device)
                    # get_encodings 
                    vae_path = f"vae_models/{args.env_id}/model.pth" 
                    state_c = get_encodings(state_c, vae_path, model)
                    state_c = np.squeeze(state_c, axis=0)
                else:
                    state_c = move[0]
                
                search_term = np.concatenate((state_c, traj[move_idx][1]))
                for ap in range(chain_length-1): # This just finds the chain
                    search_term = np.concatenate((search_term, traj[move_idx+ap+1][1])) # +ap+1 because the range will start from 0
                    
                # perform search here
                
                # search_term = torch.tensor(search_term)
                # search_term = search_term.float()
                # search_term = search_term.to(device)
            
                action_cypher=None
                for clstrr_idx in range(len(fracos_agent.clusterers)):
                    try:
                        if NN_predict:
                            with torch.no_grad():
                                predict_proba = fracos_agent.clstrr_NNs[clstrr_idx](search_term).squeeze()
                            # print("after preds = ", time.perf_counter()-NN_before)
                            cluster_label = torch.softmax(predict_proba, dim=0).argmax(dim=0)
                            # print("after softmax =", time.perf_counter()-NN_before)
                            cluster_label = cluster_label.cpu()
                            # print("after to cpu =", time.perf_counter()-NN_before)
                            strength = np.max(np.array(torch.softmax(predict_proba, dim=0).cpu()), axis=0)
                        
                        else:
                            clusterer = fracos_agent.clusterers[clstrr_idx]
                            cluster_labels, strengths = hdbscan.approximate_predict(clusterer, [search_term])
                            cluster_label = cluster_labels[0]
                            strength = strengths[0]
                        
                        if (cluster_label in fracos_agent.clusters[clstrr_idx]) and (strength >= (1 - gen_strength)):
                            # need to find where this particular option is and then decypher ... 
                            loc_label = fracos_agent.clusters[clstrr_idx].index(cluster_label)
                            action = env.action_space.n
                            for cl in range(clstrr_idx):
                                action += len(fracos_agent.clusters[cl])
                            action += loc_label
                            action_cypher = fracos_agent.cypher[action]
                    except:
                        pass # Depending on the refactoring from prim or from pre-loaded 
                        
                if action_cypher is not None:
                    if "procgen" in args.env_id:
                        refac_traj.append([move[0], action_cypher])
                    else:
                        refac_traj.append([state_c, action_cypher])
                    option_found = True
                    count = 0
                else:
                    # Need to add the old move as the new cypher action.
                    # old_fracos_agent = copy.deepcopy(fracos_agent)
                    fracos_agent.current_depth = fracos_agent.current_depth-1
                    fracos_agent.clusterers, fracos_agent.clusters, _, _, _ = fracos_agent.get_clusters()
                    fracos_agent.cypher, fracos_agent.reverse_cypher = fracos_agent.gen_cypher(fracos_agent.current_depth)
                    fracos_agent.get_clusters() # why do I need this again?
                    
                    action_revcypher = fracos_agent.reverse_cypher[tuple(move[1])]
                    
                    fracos_agent.current_depth = fracos_agent.current_depth+1
                    
                    fracos_agent.clusterers, fracos_agent.clusters, _, _, _ = fracos_agent.get_clusters()
                    fracos_agent.cypher, fracos_agent.reverse_cypher = fracos_agent.gen_cypher(fracos_agent.current_depth)
                    fracos_agent.get_clusters() # what is up with calling this at the end?
                    
                    # fracos_agent.current_depth = fracos_agent.current_depth-1
                    
                    action_cypher = fracos_agent.cypher[action_revcypher]
                    if "procgen" in args.env_id:
                        refac_traj.append([move[0], action_cypher])
                    else:
                        refac_traj.append([state_c, action_cypher])
                        
                if (move_idx == len(traj)-2) and (option_found == False): # -2 because already on -1 and then want one less
                    old_fracos_agent = copy.deepcopy(fracos_agent)
                    old_fracos_agent.current_depth = fracos_agent.current_depth-1
                    old_fracos_agent.clusterers, old_fracos_agent.clusters, _, _, _ = old_fracos_agent.get_clusters()
                    old_fracos_agent.cypher, old_fracos_agent.reverse_cypher = old_fracos_agent.gen_cypher(old_fracos_agent.current_depth)
                    old_fracos_agent.get_clusters()
                    
                    action_revcypher = old_fracos_agent.reverse_cypher[tuple(traj[move_idx+1][1])]
                    action_cypher = fracos_agent.cypher[action_revcypher]
                    if "procgen" in args.env_id:
                        refac_traj.append([move[0], action_cypher])
                    else:
                        refac_traj.append([state_c, action_cypher])
                        
        refac_trajs.append(refac_traj)
    
    return refac_trajs


def offline_cluster_compress_prim_pipeline(prim_trajectories, all_ep_rewards, failure_min, max_depth=4, chain_length=2,
                                      min_cluster_size=10, env_name="MetaGridEnv", vae_path=None,
                                      max_clusters_per_clusterer=50, gen_strength=0.33, a_pre_enc=True,
                                      NN_predict=False, max_epochs=500):
    
    # This performs the same as the offline_cluster_compress_pipeline, but can 
    # perform all building from primitive only actions. Useful for trained agents
    # such as the trained Procgen or other.
    
    trajectories = prim_trajectories
    
    if "procgen" in env_name:
        env = gym_old.make(args.env_id, num_levels=100, start_level=1, distribution_mode="easy")

    elif "highway" in env_name:
        env_in_id = env_name.split(":")[-1]
        env = gym.make(env_in_id)
        env = FlattenObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    elif env_name == "MetaGridEnv_2026":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], seed=2026)
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14])
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Josh_grid":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Josh_grid")
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Four_rooms":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Four_rooms")
        plt.imshow(env.env_master.domain)
        plt.show()
    else:
        env = gym.make(env_name)
        
    envs = SyncVectorEnv(
        [make_env(args.env_id, i, False, "None", args) for i in range(1)],
    )
    
    from fracos_agents.fracos_ppo_agent import FraCOsPPOAgent
    
    original_current_depth = copy.deepcopy(args.current_depth)
    
    for depth in range(max_depth):
        if not args.incremental:
            args.current_depth = depth
        # dimension
        action_dim = env.action_space.n
        try: # for metagrid environments
            initial_state = env.env_master.domain
        except: # for other environments
            try:
                initial_state, _ = env.reset() # for gymnasium envs
            except:
                initial_state = env.reset() # for gym envs
            
        try:
            state_dim = env.observation_space.shape[0]
        except:
            print("It is believed that the observation space is one dimensional -- determined by Discrete(xxx). If you think this is incorrect please adjust.")
            state_dim = 1
        
        fracos_agent = FraCOsPPOAgent(envs, args=args)
        all_fractures, corre_traj = create_fractures(trajectories, env_name, chain_length=chain_length
                                                    ,vae_path=vae_path, a_pre_enc=a_pre_enc, fracos_agent=fracos_agent
                                                    )
        
        print("fractures completed")
        clusterer = create_clusterer(all_fractures, MIN_CLUSTER_SIZE=min_cluster_size)
        
        print("clusters created")
        
        
        concat_fractures = sum(all_fractures, [])
        concat_trajs = sum(corre_traj, [])
        
        all_s_f = get_all_s_f_index(all_fractures, all_ep_rewards, failure_std_threshold=None,
                                    use_std=False, failure_min=failure_min)
            
        print("success and failure determined")
        
        clusterer, top_cluster, all_success_clusters,\
                ordered_cluster_pi_dict, best_clusters_list = \
                        cluster_PI_compression(clusterer, concat_fractures, all_s_f, trajectories,
                                                chain_length=chain_length, max_cluster_returns=10000, 
                                                min_PI_score = args.min_PI_score)

        print("PI compression finished")
        
        if args.style == "grid":
            save_name = env_name
        else:
            save_name = env_name+"/"+args.style
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             None, None, None, cluster_level=depth+original_current_depth, env_name=save_name)
        
        print("best clusters are : ", best_clusters_list)
        
        
        # fracos_agent = fracos_QL(state_dim, action_dim, 0.99, env_name, depth, initial_state,
        #                           vae_path=vae_path, current_depth=depth+1, gen_strength=gen_strength,
        #                           max_clusters_per_clusterer=max_clusters_per_clusterer,
        #                           chain_length=chain_length)
        
        fracos_agent = FraCOsPPOAgent(envs, args=args)

        # What about if we only create the NN based on the clusters we have available?    
        NN, model_args = create_NN_clusters(concat_fractures, list(set(clusterer.labels_)), clusterer, 
                                fracos_agent.all_possible_action_combs[depth+original_current_depth], env,
                                verbose=True, max_epochs=max_epochs, gen_str=gen_strength,
                                vae_path=vae_path)
        
        
        save_all_clusterings(clusterer, best_clusters_list, concat_fractures, concat_trajs,
                             NN, model_args, fracos_agent, cluster_level=depth+original_current_depth, env_name=save_name)
        
        ## create fracos_agent
        
        
        # fracos_agent = fracos_QL(state_dim, action_dim, 0.99, env_name, depth, initial_state,
        #                           vae_path=vae_path, current_depth=depth+1, gen_strength=gen_strength,
        #                           max_clusters_per_clusterer=max_clusters_per_clusterer,
        #                           chain_length=chain_length)
        
        args.current_depth += 1
        fracos_agent = FraCOsPPOAgent(envs, args=args)
        args.current_depth -= 1
        
        print("Refactoring trajectories")
        
        if args.debug:
            for c in fracos_agent.clusters[args.current_depth]:
                visualize_clusters_deep(args.current_depth+1, c, fracos_agent, from_offline_prim=True, env_name=args.env_id,
                                        vae=False, samples=4)
        
        if not args.incremental: # because we will generate more
            trajectories = refactor_trajectories(trajectories, fracos_agent, env_name, env,
                                  gen_strength=gen_strength, max_clusters_per_clusterer=max_clusters_per_clusterer,
                                  depth=depth, chain_length=chain_length, NN_predict=NN_predict)
            
    pass



def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_range=(0.9, 1.1)):
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], data.shape)
    return data * scale_factors

def jitter_data(data, jitter_amount=0.01):
    jitter = np.random.uniform(-jitter_amount, jitter_amount, data.shape)
    return data + jitter

def SMOTE_data(features, labels):
    smote = SMOTE(k_neighbors=3)
    # smoteT = SMOTETomek(smote=smote, random_state=42)
    augmented_features, augmented_labels = smote.fit_resample(features, labels)
    print(f"sampled from {len(features)} to {len(augmented_features)}")
    return augmented_features, augmented_labels

##### GAN PIPELINE
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Batch normalization helps stabilize training
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Updated Discriminator (you can add dropout here if needed)
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Updated GAN training function
def train_gan_for_class(X_minority, input_dim, label, epochs=5000, batch_size=32, print_interval=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    generator = Generator(input_dim=100, output_dim=input_dim).to(device)
    discriminator = Discriminator(input_dim=input_dim).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()

    # Reduce discriminator learning rate to avoid overpowering the generator
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00005)  # Reduced learning rate for the discriminator

    # Convert X_minority to a tensor
    X_minority_tensor = torch.tensor(X_minority, dtype=torch.float32)
    
    for epoch in range(epochs):
        # Sample a random batch of real minority data
        idx = np.random.randint(0, X_minority.shape[0], batch_size)
        real_samples = X_minority_tensor[idx]
        noise = torch.randn(batch_size, 100, device=device)  # Random noise for generator
        fake_samples = generator(noise)

        # Labels for real (smoothed) and fake (smoothed) data
        real_labels = torch.full((batch_size, 1), 0.9, device=device)  # Label smoothing
        fake_labels = torch.full((batch_size, 1), 0.1, device=device)  # Label smoothing

        # Add noise to inputs for discriminator
        real_samples = real_samples + 0.05 * torch.randn(real_samples.size())  # Add noise to real samples
        fake_samples = fake_samples + 0.05 * torch.randn(fake_samples.size(), device=device)  # Add noise to fake samples

        real_samples = real_samples.to(device)
        fake_samples = fake_samples.to(device)

        # Train the discriminator
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_samples), real_labels)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Calculate discriminator accuracy
        with torch.no_grad():
            real_predictions = discriminator(real_samples)
            fake_predictions = discriminator(fake_samples)
            real_acc = (real_predictions >= 0.5).float().mean().item()  # Accuracy for real samples
            fake_acc = (fake_predictions < 0.5).float().mean().item()  # Accuracy for fake samples
            discriminator_acc = 0.5 * (real_acc + fake_acc)  # Average accuracy for real and fake samples

        # Train the generator more frequently to catch up with the discriminator
        for _ in range(1):  # Train generator twice for each discriminator update
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_samples), real_labels)  # Generator wants discriminator to output 1
            g_loss.backward()  # Retain the graph to avoid freeing it
            optimizer_G.step()

        # Track progress every 'print_interval' epochs
        if epoch % print_interval == 0 or epoch == epochs - 1:
            print(f"Label: {label} | Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | D Accuracy: {discriminator_acc * 100:.2f}%")


    return generator




# Generate synthetic samples using the trained generator
def generate_synthetic_samples(generator, n_samples):
    noise = torch.randn(n_samples, 100, device=device)
    synthetic_samples = generator(noise).detach().cpu().numpy()
    return synthetic_samples

# Main pipeline function
def balance_data_with_gan(X, y):
    unique_labels = np.unique(y)
    label_counts = Counter(y)
    
    # Find the maximum count (for the majority class)
    max_count = max(label_counts.values())
    
    X_balanced = X.copy()
    y_balanced = y.copy()

    for label in unique_labels:
        X_class = X[y == label]  # Extract samples for the current class
        n_class_samples = len(X_class)
        
        if n_class_samples < max_count:
            n_synthetic_samples = max_count - n_class_samples
            
            # Train a GAN for the current class
            generator = train_gan_for_class(X_class, label=label, input_dim=X.shape[1], epochs=5000)
            
            # Generate synthetic samples
            synthetic_samples = generate_synthetic_samples(generator, n_synthetic_samples)
            synthetic_samples = synthetic_samples
            
            # Add synthetic samples to the dataset
            X_balanced = np.vstack((X_balanced, synthetic_samples))
            y_balanced = np.hstack((y_balanced, np.full(n_synthetic_samples, label)))
    
    return X_balanced, y_balanced


######


def augment_data_pipeline(features, labels):

    print("len features: ", len(features))    

    max_total_features = 5000000
    
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    max_features_per_label = int(max_total_features/len(unique))
    
    classes_to_keep = unique[counts >= 5]
    labels = np.array(labels)
    featuers = np.array(features)
    
    all_features, all_labels = balance_data_with_gan(features, labels)

    # try:
    #     over_strategy = {label: max(max_features_per_label, count) for label, count in label_counts.items()}
    #     over_sampler = SMOTE(sampling_strategy=over_strategy)
    #     all_features, all_labels = over_sampler.fit_resample(features, labels)
        
    # except:
    #     print("Error with the number of k neighbours, trying using GAN")
    #     classes_to_keep = unique[counts >= 5]
    #     labels = np.array(labels)
    #     featuers = np.array(features)
        
    #     all_features, all_labels = balance_data_with_gan(features, labels)
    
    
    return all_features, all_labels


def relabel_clusters(labels, best_labels):
    # Create a dictionary mapping each best label to its index
    label_mapping = {label: idx for idx, label in enumerate(best_labels)}
    
    # The value for labels not in best_labels will be len(best_labels)
    default_label = len(best_labels)
    
    # Apply the relabeling
    relabeled = [label_mapping.get(label, default_label) for label in labels]
    
    return relabeled


def NN_approx_init_states(clusterer, best_clusters_list, concat_fractures, obs_len, num_epochs=10, class_type="binary"):
    if class_type == "binary":
        labels = clusterer.labels_
        best_clusters_list = best_clusters_list[:args.max_clusters_per_clusterer]
        match_label_indices = np.where(np.isin(labels, best_clusters_list))[0]
        non_match_label_indices = np.where(~np.isin(labels, best_clusters_list))[0]
        
        concat_fractures = np.array(concat_fractures)
        match_fracs = concat_fractures[match_label_indices]
        non_match_fracs = concat_fractures[non_match_label_indices]
        
        match_obs_only = match_fracs[:, :obs_len]
        non_match_obs_only = non_match_fracs[:, :obs_len]
        
        match_obs_labels = np.ones(len(match_obs_only))
        non_match_obs_labels = np.zeros(len(non_match_obs_only))
        
        all_obs = np.concatenate((match_obs_only, non_match_obs_only))
        all_labels = np.concatenate((match_obs_labels, non_match_obs_labels))
        
        # we now want to augment our data
        features, labels = augment_data_pipeline(all_obs, all_labels)
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        
        # Create a TensorDataset
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        print("debug 1")
    
        model = DefaultInitClassifier(obs_len).to(device)
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            train_losses = []
            for batch_features, batch_labels in train_dataloader:
                # Forward pass
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels.float())
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_losses):.4f}')
        
            # Evaluation on training set
            if epoch % 5:
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    train_preds = []
                    train_targets = []
                    for batch_features, batch_labels in train_dataloader:
                        outputs = model(batch_features).squeeze()
                        preds = (outputs >= 0.5).int()
                        train_preds.extend(preds.cpu().numpy())
                        train_targets.extend(batch_labels.cpu().numpy())
                    train_accuracy = accuracy_score(train_targets, train_preds)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}')
                
                # Evaluation on test set
                with torch.no_grad():
                    test_preds = []
                    test_targets = []
                    for batch_features, batch_labels in test_dataloader:
                        # print(len(batch_features))
                        outputs = model(batch_features).squeeze()
                        preds = (outputs >= 0.5).int()
                        test_preds.extend(preds.cpu().numpy())
                        test_targets.extend(batch_labels.cpu().numpy())
                    test_accuracy = accuracy_score(test_targets, test_preds)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}')
                
        return model
    
    elif class_type == "multi":
        concat_fractures = np.array(concat_fractures)
        obs_only = concat_fractures[:, :obs_len]
        labels = clusterer.labels_
        
        labels = relabel_clusters(labels, best_clusters_list)
        
        # max_label = labels.max().item()
        # next_label = max_label + 1
    
        # Reassign -1 to the next available label number
        # labels[labels == -1] = next_label
        
        # features_tensor_pre = torch.tensor(obs_only, dtype=torch.float32).to(device)
        # labels_tensor_pre = torch.tensor(labels, dtype=torch.long).to(device)
        
        # dataset_pre_aug = TensorDataset(features_tensor_pre, labels_tensor_pre)
        
        
        features, labels = augment_data_pipeline(obs_only, labels)
        # features = obs_only
        
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        
        num_classes = len(np.unique(labels))
        
        
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        
        model = MultiClassClassifier(obs_len, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean', label_smoothing=0.1)  # Cross Entropy Loss for multi-class classification
        # criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss for multi-class classification
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

        
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        
        
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            train_losses = []
            for batch_features, batch_labels in train_dataloader:
                # Forward pass
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                
            scheduler.step(np.mean(train_losses))
            
            if epoch % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_losses):.4f}')
            
            # Evaluation on training set
            
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    train_preds = []
                    train_targets = []
                    for batch_features, batch_labels in train_dataloader:
                        outputs = model(batch_features)
                        _, preds = torch.max(outputs, 1)
                        train_preds.extend(preds.cpu().numpy())
                        train_targets.extend(batch_labels.cpu().numpy())
                    train_accuracy = accuracy_score(train_targets, train_preds)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.4f}')
                
                # Evaluation on test set
                with torch.no_grad():
                    test_preds = []
                    test_targets = []
                    for batch_features, batch_labels in test_dataloader:
                        outputs = model(batch_features)
                        _, preds = torch.max(outputs, 1)
                        test_preds.extend(preds.cpu().numpy())
                        test_targets.extend(batch_labels.cpu().numpy())
                    test_accuracy = accuracy_score(test_targets, test_preds)
                    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}')
                
        return model
    

def approx_policy_cluster_compress(prim_trajectories, all_ep_rewards, failure_min, chain_length=2,
                                      min_cluster_size=10, env_name="MetaGridEnv", vae_path=None,
                                      max_clusters_per_clusterer=50):
    
    # This performs the same as the offline_cluster_compress_pipeline, but can 
    # perform all building from primitive only actions. Useful for trained agents
    # such as the trained Procgen or other.
    
    trajectories = prim_trajectories
    print("num trajs are ", len(trajectories))
    
    if "procgen" in env_name:
        env = gym_old.make(args.env_id, num_levels=100, start_level=1, distribution_mode="easy")

    elif "highway" in env_name:
        env_in_id = env_name.split(":")[-1]
        env = gym.make(env_in_id)
        env = FlattenObservationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

    elif env_name == "MetaGridEnv_2026":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], seed=2026)
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14])
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Josh_grid":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Josh_grid")
        plt.imshow(env.env_master.domain)
        plt.show()
    elif env_name == "MetaGridEnv_Four_rooms":
        env = gym.make("MetaGridEnv/metagrid-v0", domain_size=[14,14], style="Four_rooms")
        plt.imshow(env.env_master.domain)
        plt.show()
    else:
        env = gym.make(env_name)
        
    envs = SyncVectorEnv(
        [make_env(args.env_id, i, False, "None", args) for i in range(1)],
    )
    
    from fracos_agents.fracos_ppo_agent import FraCOsPPOAgent
    
    original_current_depth = copy.deepcopy(args.current_depth)
    
    # dimension
    action_dim = env.action_space.n
    try: # for metagrid environments
        initial_state = env.env_master.domain
    except: # for other environments
        try:
            initial_state, _ = env.reset() # for gymnasium envs
        except:
            initial_state = env.reset() # for gym envs
        
    try:
        state_dim = env.observation_space.shape[0]
    except:
        print("It is believed that the observation space is one dimensional -- determined by Discrete(xxx). If you think this is incorrect please adjust.")
        state_dim = 1
    
    fracos_agent = FraCOsPPOAgent(envs, args=args)
    all_fractures, corre_traj = create_fractures(trajectories, env_name, chain_length=chain_length
                                                ,vae_path=vae_path, a_pre_enc=True, fracos_agent=fracos_agent
                                                )
    
    print("fractures completed")
    
    
    
    concat_fractures = sum(all_fractures, [])
    concat_trajs = sum(corre_traj, [])
    
    #clusterer = create_clusterer(all_fractures, MIN_CLUSTER_SIZE=min_cluster_size)
    clusterer = simple_clusterer(concat_fractures)
    print("clusters created")
    
    all_s_f = get_all_s_f_index(all_fractures, all_ep_rewards, failure_std_threshold=None,
                                use_std=False, failure_min=failure_min)
        
    print("success and failure determined")
    
    clusterer, top_cluster, all_success_clusters,\
            ordered_cluster_pi_dict, best_clusters_list = \
                    cluster_PI_compression(clusterer, concat_fractures, all_s_f, trajectories,
                                            chain_length=chain_length, max_cluster_returns=10000, 
                                            min_PI_score = args.min_PI_score)

    best_clusters_list2 = best_clusters_list[:args.max_clusters_per_clusterer] # to make the NN init better less likely to just choose top ones
    best_clusters_list = best_clusters_list[:args.max_clusters_per_clusterer]

    if args.vae:
        obs_shape = args.vae_latent_shape
    else:
        obs_shape = all_trajectories[0][0][0].shape[0]

    
    # predict the init states using augmentation etc
    model = NN_approx_init_states(clusterer, best_clusters_list2, concat_fractures,
                                  obs_shape, num_epochs=args.NN_epochs, class_type="multi")

    best_clusters_list = [i for i in range(len(best_clusters_list))] # becasue of the relabelling in NN_approx...
    
    ## Then we need to save this initiation NN somewhere sensible with model parameters
    
    save_all_clusterings(clusterer, best_clusters_list, concat_fractures, 
                         concat_trajs, None, None, None, args.current_depth, env_name)
    
    # save our init NN
    os.makedirs(f"fracos_clusters/{args.env_id}/a/{args.current_depth}", exist_ok=True)
    
    torch.save(model.state_dict(), f"fracos_clusters/{args.env_id}/a/{args.current_depth}/initiation.pth")
    # if multi:
        # if using all others as one label
    pickle.dump([obs_shape, len(best_clusters_list2)+1, best_clusters_list[:args.max_clusters_per_clusterer]], open(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/init_args.pkl", "wb"))
    # if binary:
    # pickle.dump(obs_shape, open(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/init_args.pkl", "wb"))
    
    ### Load our current trained agent and then save this as the policy to follow for the specified chain length if activated
    
    
    
    ## We then need to save the trajectory policy and the arguments somewhere sensible
    # (this needs to be in the gen trajectory file)


class simple_clusterer:
    def __init__(self, fractures):
        self.fractures = fractures
        self.labels_ = []
        self.obs_ = []
        
        grouped_obs = defaultdict(list)
        acts_to_key_dict = defaultdict(list)
        counter = 0
        for entry in fractures:
            key = tuple(entry[args.vae_latent_shape:])
            if key not in grouped_obs:
                acts_to_key_dict[key] = counter
                counter += 1
            grouped_obs[key].append(entry[:args.vae_latent_shape])
            self.labels_.append(acts_to_key_dict[key])
        
        self.labels_ = np.asarray(self.labels_)
            


if __name__ == "__main__":
    
    args = tyro.cli(Args)
    
    try:
        register( id="MetaGridEnv/metagrid-v0",
                  entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")

    except:
        print("MetaGridEnv already registered, skipping")
        
    
    # if not args.incremental:
    #     remove_everything_in_folder(f"fracos_clusters/{args.env_id}")
    
    
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
    
    print("traj dir is ", traj_path)
    
    all_trajectories = traj_content
    all_ep_rewards = rew_content
    
    # print("rew content is ", rew_content)
    
    if args.rm_fail:
        success_idxs = [index for index, value in enumerate(all_ep_rewards) if value > args.failure_min]
        all_trajectories = [all_trajectories[idx] for idx in success_idxs]
        all_ep_rewards = [all_ep_rewards[idx] for idx in success_idxs]
    
    
    
    
    approx_policy_cluster_compress(all_trajectories, all_ep_rewards, failure_min=args.failure_min,
                                   chain_length=args.chain_length, min_cluster_size=args.min_cluster_size,
                                   env_name=args.env_id, vae_path=None,
                                   max_clusters_per_clusterer=args.max_clusters_per_clusterer)
    
    ## FINISHED OF (2)
    
    ######### BELOW WILL TEST our visulaization of clusters in depth ########
    
    # # ## ALL SETTING UP THE AGENT AND END ###The only difference of why this would be worse is because the the clusters are not accurate enough? or there aren’t enough useful clusters? We should create more trajectories and more depth? We still see some benefit but it isn’t as pronounced as before.
    # from method.fracos_PPO import fracos_PPO
    # MAX_DEPTH = 2
    # CURRENT_DEPTH = 2
    # CHAIN_LENGTH = 2
    
    # try:
    #     register( id="MetaGridEnv/metagrid-v0",
    #       entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")
    # except:
    #     print("MetaGridEnv is already registered")
    
    # if env_name == "MetaGridEnv":
    #     env = gym.make(env_name+"/metagrid-v0", domain_size=[14,14])
    # else:
    #     env = gym.make(env_name)
        
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # lr_actor = 0.0003       # learning rate for actor network
    # lr_critic = 0.001       # learning rate for critic network
    # K_epochs = 40               # update policy for K epochs
    # eps_clip = 0.2              # clip parameter for PPO
    # gamma = 0.99                # discount factor
    # max_clusters_per_clusterer=50
    # gen_strength=0.1 
        
    # fracos_agent = fracos_PPO(state_dim, action_dim, lr_actor,
    #                                   lr_critic, gamma, K_epochs, eps_clip,
    #                                   env_name, MAX_DEPTH, chain_length=CHAIN_LENGTH,
    #                                   max_clusters_per_clusterer=max_clusters_per_clusterer,
    #                                   gen_strength=gen_strength, current_depth=CURRENT_DEPTH,
    #                                   vae_path=None)
    
    
    # visualize_clusters_deep(CURRENT_DEPTH, 58
    #                ,
    #                fracos_agent, vae=False, samples=5, MAX_DEPTH=MAX_DEPTH,
    #               chain_length=2, from_offline_prim=True, env_name=env_name)
    