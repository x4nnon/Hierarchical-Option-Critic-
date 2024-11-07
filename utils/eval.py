#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:11:19 2023

@author: x4nno
"""
import sys
sys.path.append("/home/x4nno/Documents/PhD/FRACOs_v4.1")

import numpy as np

import os 
from matplotlib import pyplot as plt
import pandas as pd 
import random
import torch
import gym

device = torch.device('cpu')

def create_trajectories(method, num_of_trajs, model_path, env, max_ep_length, obs_space=False,
                        evalu=True):
    method.load(model_path)
    trajectories = []
    ep_rewards = []
    discrete_search_cache = {}
    for ep in range(1, num_of_trajs):
        
        trajectory = []
        ep_reward = 0
        try:
            state, _ = env.reset()
        except:
            state, _ = env.reset()
            
        time_step = 0
        while time_step < max_ep_length:
            if evalu == True:
                action, discrete_search_cache = method.select_action_eval(state, env, discrete_search_cache, method)
            else:
                action, discrete_search_cache = method.select_action(state, env, discrete_search_cache)
            
            cyph_action = method.cypher[action]
            trajectory.append([state, cyph_action])
            
            env, method.buffer, reward, time_step, discrete_search_cache, steps_rewards, done, state\
                = method.take_recursive_cluster_action(action, env, state,
                                                                 False, time_step, method.buffer,
                                                                 discrete_search_cache=discrete_search_cache,
                                                                 steps_rewards=[]
                                                                 )
            
            ep_reward += reward
            
            if done:
                break
        print(ep , " of " , num_of_trajs, " --- REW --- ", ep_reward)
        trajectories.append(trajectory)
        ep_rewards.append(ep_reward)
        
    return trajectories, ep_rewards

def create_even_trajectories(method, max_timesteps, model_path, env, max_ep_length, obs_space=False,
                        evalu=True, rew_gen_meta=True, one_level=False):
    if model_path is not None:
        method.load(model_path)
    trajectories = []
    ep_rewards = []
    discrete_search_cache = {}
    total_timestep = 0
    while total_timestep < max_timesteps:
        
        trajectory = []
        ep_reward = 0
        if one_level:
            try:
                state, _ = env.reset(start_same=True)
            except:
                state, _ = env.reset(start_same=True)
        elif rew_gen_meta:
            try:
                state, _ = env.reset(start_same=False, total_random=True) # total random actually refers to rewards only not state
            except:
                state, _ = env.reset(start_same=False, total_random=True)
        else:
            try:
                state, _ = env.reset()
            except:
                state, _ = env.reset()
                
        # for debugging
        # plt.imshow(env.env_master.domain)
        # plt.show()
            
        time_step = 0
        while time_step < max_ep_length:
            if evalu == True:
                action, discrete_search_cache = method.select_action_eval(state, env, discrete_search_cache, argmax=True)
            else:
                action, discrete_search_cache = method.select_action(state, env, discrete_search_cache)
            
            cyph_action = method.cyphers[method.current_depth][action]
            trajectory.append([state, cyph_action])
            
            env, method.buffer, reward, time_step, discrete_search_cache, steps_rewards, done, state\
                = method.take_recursive_cluster_action(action, env, state,
                                                                 False, time_step, method.buffer,
                                                                 discrete_search_cache=discrete_search_cache,
                                                                 steps_rewards=[]
                                                                 )
            
            ep_reward += reward
            
            if done:
                break
        total_timestep += time_step
        #print(ep , " of " , num_of_trajs, " --- REW --- ", ep_reward)

        trajectories.append(trajectory)
        ep_rewards.append(ep_reward)
    
    print( "------", len(trajectories))
    return trajectories, ep_rewards

def create_ind_opt_trajectories(method, num_of_trajs, env_name,
                                max_ep_length, max_train_steps, 
                                MAX_DEPTH, CURRENT_DEPTH, chain_length=2,
                                print_freq=1000, log_freq=500, shuffle=False,
                                save_model_freq=1000, update_timestep=1000,
                                obs_space=False, verbose=False, vae_path=None,
                                max_clusters_per_clusterer=100, gen_strength=0.01,
                                random_seed=0, optish_min=4.55, argmax_end=False,
                                passed_env=None, reward_gen_only=False,
                                update_episodes=5, update_timestep_or_ep="ep", min_training_steps=12000,
                                gen_learn=False, gen_trajectories=None, method_name="tabular_q"):
    """This will freeze an environment and train on it for max_ep_length up until max_train_steps
    Then it will take the optimal trajectory simlar to argmax_episodes, this is repeated for num_of_trajs,
    times."""
    
    if random_seed:
        random.seed(random_seed)
        random_seeds = [random.randint(1,10000) for ep in range(0, num_of_trajs)]
    else:
        random_seeds = [0 for ep in range(0, num_of_trajs)]

    # import within function to avoid circular errors
    from train.fracos_all import train_pipe

    trajectories = []
    ep_rewards = []
    envs = []
    discrete_search_cache = {}
    for ep in range(1, num_of_trajs):
        print("starting ep ", ep, "of ", num_of_trajs)
        ## First we want to learn - and by default this will create a random
        ## environments, choose shuffle=False to keep them the same through learning
        ## we set save model freq to max_train_steps*2 so that it never happens
        
        save_model_freq = int(1e5)      # save model frequency (in num timesteps)
        eval_eps = 1 # for debug set low # for training set higher to ensure converge.
        
        
        discrete_search_cache, env, method = train_pipe(env_name, max_ep_length, max_train_steps,
                                        print_freq, log_freq, save_model_freq, eval_eps,
                                        update_timestep, MAX_DEPTH, chain_length=chain_length,
                                        max_clusters_per_clusterer=max_clusters_per_clusterer,
                                        gen_strength=gen_strength, discrete_search_cache={},
                                        current_depth=CURRENT_DEPTH, shuffle=shuffle,
                                        verbose=verbose, vae_path=vae_path, gen_traj=True,
                                        random_seed=random_seeds[ep], optish_min=optish_min,
                                        argmax=argmax_end, passed_env=passed_env,
                                        reward_gen_only=reward_gen_only, 
                                        fixed=True, # This keeps the environment the same when we are learning.
                                        update_timestep_or_ep=update_timestep_or_ep,
                                        update_episodes=update_episodes, method_name=method_name,
                                        min_training_steps=min_training_steps
                                        )
        
        
        if gen_learn:
            shuffle=True # override from function
            for t in range(gen_trajectories):
                rewards, discrete_search_cache, trajectory = argmax_episodes(method, env,
                                                                             1, max_ep_length,
                                                                             discrete_search_cache,
                                                                             shuffle=shuffle,
                                                                             verbose=verbose,
                                                                             env_name = env_name,
                                                                             random_seed=random_seeds[ep],
                                                                             argmax=True, method_name=method_name)
                
                trajectory = trajectory[0]
                ep_rewards.append(rewards)
                trajectories.append(trajectory)
                envs.append(env)

                print("ep : ", ep)
                print(" --- reward : ", rewards)
            return trajectories, ep_rewards, envs
                
        rewards, discrete_search_cache, trajectory = argmax_episodes(method, env,
                                                                     1, max_ep_length,
                                                                     discrete_search_cache,
                                                                     shuffle=shuffle,
                                                                     verbose=verbose,
                                                                     env_name = env_name,
                                                                     random_seed=random_seeds[ep],
                                                                     argmax=True, method_name=method_name)
        
        trajectory = trajectory[0]
        ep_rewards.append(rewards)
        trajectories.append(trajectory)
        envs.append(env)

        print("ep : ", ep)
        print(" --- reward : ", rewards)
        
        
    return trajectories, ep_rewards, envs

def create_random_trajectories(method, num_of_trajs, model_path, env, max_ep_length, obs_space=False):
    method.load(model_path)
    trajectories = []
    ep_rewards = []
    discrete_search_cache = {}
    for ep in range(1, num_of_trajs):
        trajectory = []
        ep_reward = 0
        state, _ = env.reset()
        time_step = 0
        while time_step < max_ep_length:
            action = random.choice(range(method.total_action_dims))
            discrete_search_cache = method.initial_search(state, discrete_search_cache)
            env, method.buffer, reward, time_step, discrete_search_cache, steps_rewards, done, state\
                = method.take_recursive_cluster_action(action, env, state,
                                                                 False, time_step, method.buffer,
                                                                 discrete_search_cache=discrete_search_cache
                                                                 , steps_rewards=[])
            
            ep_reward += reward
            cyph_action = method.cyphers[method.current_depth][action]
            trajectory.append([state, cyph_action])
            if done:
                break
            
            time_step += 1
            
        trajectories.append(trajectory)
        ep_rewards.append(ep_reward)
        
    return trajectories, ep_rewards


def argmax_episodes(method, env, num_of_trajs, max_ep_length,
                    discrete_search_cache, shuffle=True,
                    verbose=False, method_name="PPO", argmax=True, env_name=None,
                    random_seed=0):
    ep_rewards = []
    trajectories = []
    for ep in range(1, num_of_trajs+1):
        trajectory = []
        ep_reward = 0
        
        if (shuffle) & ("MetaGrid" in env_name):
            state, _ = env.reset(start_same=False, total_random=False)
        elif shuffle:
            state, _ = env.reset()

                
        else:
            if "MetaGrid" not in env_name:
                try:
                    env.seed(random_seed)
                    state, _ = env.reset()
                except:
                    state, _ = env.reset(seed=random_seed)
                
            else:
                state, _ = env.reset(start_same=True)

        
        time_step = 0
        
        option_count = 0
        while time_step < max_ep_length:
            # change to select_action if needed for tracking
            if method.name == "PPO":
                action, discrete_search_cache = method.select_action_eval(state, env, discrete_search_cache,
                                                                          argmax=argmax)
            elif method.name == "tabular_q":
                action, discrete_search_cache = method.select_action(state, env, discrete_search_cache,
                                                                          argmax=argmax)
            else:
                print("Only PPO and tabular_q are implemented for argmax episodes")
            
            cyph_action = method.cyphers[method.current_depth][action]
            if np.all(cyph_action[:4] == 0):
                option_count += 1
            
            trajectory.append([state, cyph_action])
            
            if method_name == "PPO":
                env, method.buffer, reward, time_step, discrete_search_cache, steps_rewards, done, state\
                    = method.take_recursive_cluster_action(action, env, state,
                                                                     False, time_step, method.buffer,
                                                                     discrete_search_cache=discrete_search_cache,
                                                                     evalu=True, steps_rewards = []
                                                                     )
                    
            else:
                env, reward, time_step, discrete_search_cache, steps_rewards, done, state\
                    = method.take_recursive_cluster_action(action, env, state,
                                                                     False, time_step,
                                                                     discrete_search_cache=discrete_search_cache
                                                                     ,evalu=True, 
                                                                     steps_rewards = []
                                                                     )
            # if verbose:
            #     env.render()
            ep_reward += reward
            
            
            if done:
                break
        if verbose:
            print("argmax reward is : ", ep_reward)
            print("number of options is : ", option_count)
        trajectories.append(trajectory)
        ep_rewards.append(ep_reward)
        
        # Why is this here below?
        
        if shuffle:
            state, _ = env.reset()
        else:
            if "MetaGrid" not in env_name:
                try:
                    env.seed(random_seed)
                    state, _ = env.reset()
                except:
                    state, _ = env.reset(seed=random_seed)
            else:
                state, _ = env.reset(start_same=True)

    return ep_rewards, discrete_search_cache, trajectories
        

def produce_eval_graphs(MAX_DEPTH, env_name, method_name="PPO", log_dir=None):
    
    for i in range(MAX_DEPTH+1):
        first = True
        if log_dir is None:
            log_dir = "logs/fracos_{}_logs".format(method_name) + '/' + env_name + '/'
        log_eval_dir = log_dir + "/eval_log/cluster_depth_{}".format(i)
        
        print(log_eval_dir)
        file_count = 0
        for file in os.listdir(log_eval_dir):
            file_path = log_eval_dir + '/' + file

            if first:
                df_concat = pd.read_csv(file_path)
                first = False
            else:
                
                df = pd.read_csv(file_path)
                df_concat = pd.concat((df_concat, df), axis=1)
            file_count += 1
        
        df_concat = df_concat.dropna()
        x = df_concat["timestep"]
        try:
            x_means = x.mean(axis=1)
        except:
            print("not enough files for mean or std")
            x_means = x
        y = df_concat["average_reward_over_eps"] # change number to however many we avg over
        try:
            y_means = y.mean(axis=1)
            y_sems = y.sem(axis=1)
            upper_b = y_means + y_sems
            lower_b = y_means - y_sems
        except:
            print("not enough files for mean or std")
            y_means = y
        
        x_means = np.array(x_means)
        y_means = np.array(y_means)
        
        # !!! to do: error bars for stds - fill with transluscent
                
        if i == 0:
            plt.plot(x_means, y_means, label="Primitive")
        else:
            plt.plot(x_means, y_means, label="FraCOs_depth = {}".format(i))
        try:
            plt.fill_between(x_means, lower_b, upper_b, alpha=0.2)
        except:
            print("not enough files for mean or std")
            
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Timesteps")
    plt.ylabel("Averaged eval reward")
    plt.title("{} eval rewards, on {} seperate training runs".format(env_name, file_count))
    plt.show()
    
def produce_OFU_graphs(list_of_dirs, env_name, labels):
    
    count = 0
    for log_eval_dir in list_of_dirs:
        first=True
        file_count = 0
        for file in os.listdir(log_eval_dir):
            file_path = log_eval_dir + '/' + file

            if first:
                df_concat = pd.read_csv(file_path)
                first = False
            else:
                
                df = pd.read_csv(file_path)
                df_concat = pd.concat((df_concat, df), axis=1)
            file_count += 1
        
        df_concat = df_concat.dropna()
        x = df_concat["timestep"]
        try:
            x_means = x.mean(axis=1)
        except:
            print("not enough files for mean or std")
            x_means = x
        y = df_concat["average_reward_over_eps"] # change number to however many we avg over
        try:
            y_means = y.mean(axis=1)
            y_sems = y.sem(axis=1)
            upper_b = y_means + y_sems
            lower_b = y_means - y_sems
        except:
            print("not enough files for mean or std")
            y_means = y
        
        
        # !!! to do: error bars for stds - fill with transluscent
        # if "no_OFU" in log_eval_dir:
        #     plt.plot(x_means, y_means, label="cluster_depth_no_OFU = {}".format(1))
        # else:
        #     plt.plot(x_means, y_means, label="cluster_depth_OFU = {}".format(1))
        # try:
        #     plt.fill_between(x_means, lower_b, upper_b, alpha=0.2, label="1std error")
        # except:
        #     print("not enough files for mean or std")
        
        plt.plot(x_means, y_means, label=labels[count])
        plt.fill_between(x_means, lower_b, upper_b, alpha=0.2)
            
        count += 1
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Timesteps")
    plt.ylabel("Averaged eval reward")
    plt.title("{} Eval rewards, over {} seperate training runs".format(env_name, file_count))
    plt.show()
    
def produce_learning_graphs(MAX_DEPTH, env_name, method_name="PPO", log_dir=None):

    # ONLY TRUE FOR one level of clustering need to change this in the future    
    
    ##########
    
    for i in range(MAX_DEPTH+1):
        first = True
        if log_dir is None:
            log_dir = "logs/fracos_{}_logs".format(method_name) + '/' + env_name + '/'
        log_eval_dir = log_dir + "/eval_log/cluster_depth_{}".format(i)
        
        file_count = 0
        for file in os.listdir(log_eval_dir):
            file_path = log_dir + file.split("_eval")[0] + file.split("_eval")[1]
            
            if first:
                df_concat = pd.read_csv(file_path)
                first = False
            else:
                
                df = pd.read_csv(file_path)
                df_concat = pd.concat((df_concat, df), axis=1)
            file_count += 1
        
        df_concat = df_concat.dropna()
        x = df_concat["epoch"]
        try:
            x_means = x.mean(axis=1)
        except:
            print("not enough files for mean or std")
            x_means = x
        y = df_concat["reward"] # change number to however many we avg over
        try:
            y_means = y.mean(axis=1)
            y_sems = y.sem(axis=1)
            upper_b = y_means + y_sems
            lower_b = y_means - y_sems
        except:
            print("not enough files for mean or std")
            y_means = y
        
        x_means = np.array(x_means)
        y_means = np.array(y_means)
        
        if i == 0:
            plt.plot(x_means, y_means, label="Primitive")
        else:
            plt.plot(x_means, y_means, label="FraCOs_depth = {}".format(i))
        try:
            plt.fill_between(x_means, lower_b, upper_b, alpha=0.2)
        except:
            print("not enough files for mean or std")
            
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel("Epoch")
    plt.ylabel("Total epoch rewards")
    plt.title("{} epoch rewards, on {} seperate training runs".format(env_name, file_count))
    plt.show()
    
    
def produce_option_graphs(MAX_DEPTH, env_name, method_name="PPO"):
    # ONLY TRUE FOR one level of clustering need to change this in the future   
    # Also only counts total options used and not tracking how often each option is used. 

    first = True
    log_dir = "logs/fracos_{}_logs".format(method_name) + '/' + env_name + '/'
    
    print(log_dir)
    # file_count = 0
    # for file in os.listdir(log_dir):
    #     print(file)
    #     if file == "eval_log":
    #         pass
    #     elif int(file[-5]) % 2 == 0:
    #         print(int(file[-5]) % 2)
    #         file_path = log_dir + '/' + file
    
    #         if first:
    #             df_concat = pd.read_csv(file_path)
    #             first = False
    #         else:
                
    #             df = pd.read_csv(file_path)
    #             df_concat = pd.concat((df_concat, df), axis=1)
    #         file_count += 1
    # print(file_count)
    # df_concat = df_concat.dropna()
    # x = df_concat["options"]
    # x_means = x.mean(axis=1)
    # y = df_concat["reward"] # change number to however many we avg over
    # y_means = y.mean(axis=1)
    # y_stds = y.std(axis=1)
    
    # upper_b = y_means + y_stds
    # lower_b = y_means - y_stds
    # # !!! to do: error bars for stds - fill with transluscent 
            
    # plt.scatter(x_means, y_means, label="cluster_depth = {}".format(0))
    # # plt.fill_between(x_means, lower_b, upper_b, alpha=0.2, label="1std error")
    
    # plt.errorbar(x_means, y_means, yerr=y_stds, label="1std",
    #              fmt="o")
    
    first = True
    file_count = 0
    for file in os.listdir(log_dir):
        if file == "eval_log":
            pass
        elif int(file[-5]) % 2 != 0:
            file_path = log_dir + '/' + file
    
            if first:
                df_concat = pd.read_csv(file_path)
                df_concat = df_concat.iloc[-1:]
                first = False
            else:
                df = pd.read_csv(file_path)
                df = df.iloc[-1:]
                df_concat = pd.concat((df_concat, df), axis=0)
            file_count += 1
        
    df_concat = df_concat.dropna()
    x = df_concat["options"]
    x_means = x.mean()
    y = df_concat["reward"] # change number to however many we avg over
    y_means = y.mean()
    y_sems = y.sem()
    
    upper_b = y_means + y_sems
    lower_b = y_means - y_sems
    # !!! to do: error bars for stds - fill with transluscent 
            
    plt.scatter(x, y, label="cluster_depth = {}".format(1))
    # plt.fill_between(x_means, lower_b, upper_b, alpha=0.2, label="1std error")
        
    # plt.errorbar(x_means, y_means, yerr=y_stds, label="1std",
    #              fmt="o")
    
    plt.legend()
    plt.xlabel("Number of options used in Epoch")
    plt.ylabel("summed rewards in epoch")
    plt.title("Averaged rewards compared with options used for {} seperate training runs".format(file_count))
    plt.show()
   
if __name__ == "__main__":
    MAX_DEPTH = 2
    env_name = "MetaGridEnv"
    # env_name = "CartPole-v1"
    # produce_learning_graphs(MAX_DEPTH, env_name)
    # produce_eval_graphs(MAX_DEPTH, env_name)
    # produce_option_graphs(MAX_DEPTH, env_name)
    
    # list_of_dirs = ["/home/x4nno/Documents/PhD/FRACOs_v3/archive/eval_logs/Simsek_gen07/MetaGridEnv/eval_log/cluster_depth_0",
    #                 "/home/x4nno/Documents/PhD/FRACOs_v3/archive/eval_logs/Simsek_gen01/MetaGridEnv/eval_log/cluster_depth_2",
    #                 "/home/x4nno/Documents/PhD/FRACOs_v3/archive/eval_logs/Simsek_gen03/MetaGridEnv/eval_log/cluster_depth_2",
    #                 "/home/x4nno/Documents/PhD/FRACOs_v3/archive/eval_logs/Simsek_gen05/MetaGridEnv/eval_log/cluster_depth_2",
    #                 "/home/x4nno/Documents/PhD/FRACOs_v3/archive/eval_logs/Simsek_gen07/MetaGridEnv/eval_log/cluster_depth_2"]
    
    # labels = ["primitive", "G=0.1", "G=0.3", "G=0.5", "G=0.7"]
    
    # produce_OFU_graphs(list_of_dirs, env_name, labels)
    
    log_dir = "/home/x4nno/Documents/PhD/FRACOs_v4.1/archive/MetaGrid2026/fracos_tabular_q_logs/MetaGridEnv/"
    
    produce_eval_graphs(MAX_DEPTH, env_name, "PPO", log_dir=log_dir)
    produce_learning_graphs(MAX_DEPTH, env_name, "PPO", log_dir=log_dir)