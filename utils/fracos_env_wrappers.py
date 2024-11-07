# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
import sys

sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno/Documents/PhD/FRACOs_a")

sys.path.append("/app/MetaGridEnv/MetaGridEnv")
sys.path.append("/app/FRACOs_v6_ppo_hex_only")
sys.path.append("/home/x4nno_desktop/Documents/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno_desktop/Documents/FRACOs_a")

from gym import Wrapper

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
import copy
from gym.envs.registration import register 
from matplotlib import pyplot as plt
from torch.distributions.categorical import Categorical

register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")

from utils.sync_vector_env import SyncVectorEnv
from fracos_agents.fracos_ppo_agent import FraCOsPPOAgent
import time
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_agent_location(array):
    loc = np.where(array == 2)
    try:
        return [loc[0][0], loc[1][0]]
    except:
        print("can't find the agent ...")
        return [None, None]
    
    
def plt_em(env):
    plt.imshow(env.env_master.domain)
    plt.show()


def plot_current_obs(next_ob, action):
    plt.imshow(next_ob.cpu().squeeze().permute(1,2,0)/255)
    plt.title(action)
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
    
    
class fracos_ppo_fixed_mdp_reset(Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        self.args = args
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(seed=self.args.seed)
        return observation, info
    
    
class fracos_ppo_procgen_seed(Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.env = env
        self.args = args
        
    def seed(self, seed):
        pass
        

class fracos_ppo_wrapper(Wrapper):
    def __init__(self, env, max_ep_length):
        super().__init__(env)
        self.max_ep_length = max_ep_length
        
    def fracos_step(self, action, next_ob, next_ob_np_flat, agent, total_rewards=0, 
                    total_steps_taken=0, agent_update=False, top_level=False, chain_length=2, isdone=False):

        all_info = {}
        
        # try:
        #     # this is needed for procgen. For atari we don't need this. So when doing atari we will need to change this ..
        #     if (next_ob.shape[-1] != 3):
        #         next_ob = next_ob.permute(2,0,1)
        #     ob = tuple(next_ob.view(-1).tolist())

        # except:
        #     if torch.is_tensor(next_ob):
        #         ob = tuple(next_ob.view(-1).tolist())
        #     else:
        #         ob = tuple(next_ob)
        # next_ob_byte = next_ob_np_flat.tobytes()
        
        if action not in range(agent.action_prims):
            fracos_policy_idx = int(((action-15)/agent.args.max_clusters_per_clusterer)//1)
            fracos_policy = agent.fracos_policies[fracos_policy_idx]
            
            for ac in range(chain_length):
                with torch.no_grad():
                    if isinstance(next_ob, np.ndarray):
                        next_ob = torch.tensor(next_ob).to(device)
                    if next_ob.shape[-1] == 3: # procgen comes in a weird format
                        next_ob = next_ob.permute(2,0,1)
                    if len(next_ob.shape) != 4:
                        next_ob = next_ob.unsqueeze(0)
                    next_ob = next_ob.float()
                    fracos_network = fracos_policy.network.to(device)
                    fracos_actor = fracos_policy.actor.to(device)
                    hidden = fracos_network(next_ob/255)
                    new_action_logits = fracos_actor(hidden)
                    new_action = torch.argmax(new_action_logits).item()
                    
                    # use below for debugging if needed
                    # plot_current_obs(next_ob, new_action)

                    next_ob, next_ob_np_flat, total_rewards, reward, termination, truncation, \
                        info, total_steps_taken = \
                        self.fracos_step(new_action, next_ob, next_ob_np_flat, agent, 
                                         total_rewards=total_rewards, total_steps_taken=total_steps_taken)
                        
                    all_info.update(info)
                    
                    next_done = np.logical_or(termination, truncation)
                    if next_done:
                        return next_ob, next_ob_np_flat, total_rewards, reward, termination, truncation, info, total_steps_taken
                    
                
        else:
            # b4 = time.time() # remove
            next_ob, reward, termination, truncation, info = self.env.step(action)
            next_ob_np_flat = next_ob.ravel() # because it will arrive as a np.array
            total_rewards += reward
            total_steps_taken += 1
            next_done = np.logical_or(termination, truncation)
            if next_done:
                return next_ob, next_ob_np_flat, total_rewards, reward, termination, truncation, info, total_steps_taken
            
            # after = time.time() # remove
            # print("Time taken normal = ", after-b4) # remove
            
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
        
        return next_ob, next_ob_np_flat, total_rewards, reward, termination, truncation, info, total_steps_taken
    
    
class fracos_tabularQ_wrapper(Wrapper):
    
    def __init__(self, env, max_ep_length):
        super().__init__(env)
        self.max_ep_length = max_ep_length
        
    def fracos_step(self, action, next_ob, next_ob_flat, agent, total_rewards=0, total_steps_taken=0, agent_update=True, top_level=False):
        all_info = {}
        try:
            ob = tuple(next_ob.cpu().numpy())
        except:
            ob = tuple(next_ob)
        
        prev_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
        prev_steps = total_steps_taken
        
        if action not in range(agent.action_prims):
            if agent.top_only and top_level:
                action = action - agent.total_action_dims + agent.all_unrestricted_actions # undo the previous changes
            
            if ob not in agent.discrete_search_cache:
                agent.initial_search(ob, ob)
            id_actions = tuple(agent.discrete_search_cache[ob][action])
            if isinstance(id_actions[0], np.ndarray):
                for id_action in id_actions:
                    
                    if "HRL issue" in all_info:
                        break # do not carry on if failed on this!
                    
                    for reverse_cypher in agent.reverse_cyphers:
                        if tuple(id_action) in reverse_cypher.keys():
                            id_action = reverse_cypher[tuple(id_action)]
                            break
    
                    next_ob, next_ob_flat, total_rewards, reward, termination, truncation, info, total_steps_taken = \
                        self.fracos_step(id_action, next_ob, next_ob_flat, agent, total_rewards=total_rewards, total_steps_taken=total_steps_taken)
                        
                    all_info.update(info)
                    
                    # need to exit if we have finished
                    next_done = np.logical_or(termination, truncation)
                    if next_done:
                        next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
                        next_steps = total_steps_taken
                        if agent_update:
                            agent.update(action, max(total_rewards, reward),
                                         prev_agent_loc, next_agent_loc)
                            
                        return next_ob, next_ob_flat, total_rewards, reward, termination, truncation, all_info, total_steps_taken
                        
            else:
                # returns a negative reward and our current location.
                
                # This is handled through the INFO tag to stop it happening again,
                info = {"HRL issue" : True}
                all_info.update(info)
                # equiv of the update here
                agent.discrete_search_cache[ob][action] = [None, None]
                agent.Q[tuple(find_agent_location(self.env.env_master.domain))][action] = -1000
                
                return ob, next_ob_flat, total_rewards, 0, False, None, all_info, total_steps_taken
        else:
            next_ob, reward, termination, truncation, info = self.env.step(action)
            nest_ob_flat = tuple(next_ob)
            all_info.update(info)
            total_rewards += reward
            total_steps_taken += 1
            next_done = np.logical_or(termination, truncation)
            if next_done:
                next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
                next_steps = total_steps_taken
                
                if agent_update:
                    agent.update(action, max(total_rewards, reward),
                                 prev_agent_loc, next_agent_loc)
                return next_ob, next_ob_flat, total_rewards, reward, termination, truncation, all_info, total_steps_taken
            
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
        

        if "HRL issue" in info:
            with torch.no_grad():
                agent.discrete_search_cache[ob][action] = [None, None]
        
        next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
        next_steps = total_steps_taken
        if agent_update:
            if next_steps > prev_steps:
                agent.update(action, max(total_rewards, reward),
                             prev_agent_loc, next_agent_loc)
            
        return next_ob, next_ob_flat, total_rewards, reward, termination, truncation, info, total_steps_taken
    
# class fracos_tabularQ_wrapper(Wrapper):
    
#     def __init__(self, env, max_ep_length):
#         super().__init__(env)
#         self.max_ep_length = max_ep_length
        
#     def fracos_step(self, action, next_ob, agent, total_rewards=0, total_steps_taken=0, agent_update=True):
#         all_info = {}
#         try:
#             ob = tuple(next_ob.cpu().numpy())
#         except:
#             ob = tuple(next_ob)
        
#         prev_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
#         prev_steps = total_steps_taken
        
#         if action not in range(agent.action_prims):
#             if ob not in agent.discrete_search_cache.keys():
#                 agent.initial_search(ob)
#             id_actions = tuple(agent.discrete_search_cache[ob][action])
#             if isinstance(id_actions[0], np.ndarray):
#                 for id_action in id_actions:
#                     for reverse_cypher in agent.reverse_cyphers:
#                         if tuple(id_action) in reverse_cypher.keys():
#                             id_action = reverse_cypher[tuple(id_action)]
#                             break
    
#                     next_ob, total_rewards, termination, truncation, info, total_steps_taken = \
#                         self.fracos_step(id_action, next_ob, agent, total_rewards=total_rewards, total_steps_taken=total_steps_taken)
                        
#                     all_info.update(info)
                    
#                     # need to exit if we have finished
#                     next_done = np.logical_or(termination, truncation)
#                     if next_done:
#                         next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
#                         next_steps = total_steps_taken
#                         if agent_update:
#                             agent.update(action, total_rewards/(next_steps-prev_steps),
#                                          prev_agent_loc, next_agent_loc)
                            
#                         return next_ob, total_rewards, termination, truncation, all_info, total_steps_taken
                        
#             else:
#                 # returns a negative reward and our current location.
                # fracos_ppo_fixed_mdp_reset
#                 #This is handled through the INFO tag to stop it happening again,
#                 info = {"HRL issue" : True}
#                 all_info.update(info)
#                 return ob, total_rewards, False, None, all_info, total_steps_taken
#         else:
#             next_ob, reward, termination, truncation, info = self.env.step(action)
#             all_info.update(info)
#             total_rewards += reward
#             total_steps_taken += 1
#             next_done = np.logical_or(termination, truncation)
#             if next_done:
#                 next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
#                 next_steps = total_steps_taken
#                 if agent_update:
#                     agent.update(action, total_rewards/(next_steps-prev_steps),
#                                  prev_agent_loc, next_agent_loc)
#                 return next_ob, total_rewards, termination, truncation, all_info, total_steps_taken
            
#         if self.env.episode_lengths > self.max_ep_length:
#             truncation = True
#         else:
#             truncation = False
            
#         dones = np.logical_or(termination, truncation)
#         num_dones = np.sum(dones)
#         if num_dones:
#             if "episode" in info or "_episode" in info:
#                 raise ValueError(
#                     "Attempted to add episode stats when they already exist"
#                 )
#             else:
#                 info["episode"] = {
#                     "r": np.where(dones, self.episode_returns, 0.0),
#                     "l": np.where(dones, self.episode_lengths, 0),
#                     "t": np.where(
#                         dones,
#                         np.round(time.perf_counter() - self.episode_start_times, 6),
#                         0.0,
#                     ),
#                 } 
        

#         if "HRL issue" in info:
#             with torch.no_grad():
#                 agent.discrete_search_cache[ob][action] = [None, None]
        
#         next_agent_loc = tuple(find_agent_location(self.env.env_master.domain))
#         next_steps = total_steps_taken
#         if agent_update:
#             if next_steps > prev_steps: # sometimes everything can fail and we will get a divide by 0 error.
#                 agent.update(action, total_rewards/(next_steps-prev_steps),
#                              prev_agent_loc, next_agent_loc)
            
#         return next_ob, total_rewards, termination, truncation, info, total_steps_taken
    


if __name__ == "__main__":
    pass
