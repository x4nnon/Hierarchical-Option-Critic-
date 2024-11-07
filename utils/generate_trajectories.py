#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:05:46 2024

@author: x4nno
"""

# BELOW IS THE WORKFLOW WE NEED TO FOLLOW. 

# 1. define a list of seeds

# 2. learn each of the seeds while fixing the MDP (domain, reward and state_space)

# 3. generate an eval trajectory once suitably learned

import os

import sys
sys.path.append('.')
sys.path.append('..')


from dataclasses import dataclass
import tyro
from methods import fracos_ppo, fracos_tabularQ
import random
import torch
from datetime import datetime
from utils.sync_vector_env import SyncVectorEnv
import numpy as np
import pickle
import os
from matplotlib import pyplot as plt
import ast

from fracos_agents.fracos_ppo_agent import FraCOsPPOAgent
    
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False #!!! change to True for running realtime
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "CoinRun_fracos_rerun_proc"
    """the wandb's project name"""
    wandb_entity: str = "tpcannon"
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "procgen-starpilot"
    """the id of the environment: MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun,
    atari:BreakoutNoFrameskip-v4, highway:highway-fast-v0"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True #always true
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False # this was True when gathering good results.
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.001
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    report_epoch: int = num_steps*num_envs*10
    """When to run a seperate epoch run to be reported. Make sure this is a multple of num_envs."""
    anneal_ent: bool = False
    """Toggle entropy coeff annealing"""
    domain_size: int = 14
    """The size of the metagrid domain if using metagrid"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # Below are fracos specific
    max_clusters_per_clusterer: int = 50
    """the maximum number of clusters at each hierarchy level"""
    current_depth: int = 0
    """this is the current level of hierarchy we are considering"""
    chain_length: int = 3
    """How long our option chains are"""
    NN_cluster_search: bool = True
    """Should we use NN to predict our clusters? if false will use hdbscan"""
    gen_strength: float = 0.33
    """This should be the strength of generalisation. for NN 0.1 seems good. for hdbscan 0.33"""    
    FraCOs_bias_factor: float = 1
    """How much to multiply the logit by to bias towards choosing the identified fracos"""
    FraCOs_bias_depth_anneal: bool = False
    """If True, then lower depths will have less bias factor than higher depths -- encourages searching higher depths first"""
    max_ep_length: int = 1000 # massive as default *** importatnt, for procgen this limits the eval ep lengths.
    """Max episode length"""
    fix_mdp: bool = False
    """whether the mdp should be fixed on reset. useful for generating trajectories."""
    gen_traj: int = 1 # MUST CHANGE THIS to FALSE BEFORE RUNNING ELSE WONT GET ANY RESULTS FROM EVALS!
    """ A tag which should be used if we are generating a trajectory, not for testing. """
    top_only: bool = False # not implemented yet
    """ This will cause the agent to only use the top level of abstraction and the primitives."""
    vae: bool = False ## Need to set this to false as a default
    
    vae_latent_shape: int = 10
    resnet_features: int = 0 # 1 for true and 0 for false
    
    debug: bool = False # makesure to change to False before running
    
    #env_specific
    style: str = "grid"
    proc_start: int = 1 # 0 for train. > 50 for test
    proc_num_levels: int = 100 # 50 for train. > for test
    proc_sequential=False
    max_eval_ep_len: int = 1000
    sep_evals: int = 0 # 1 for true, 0 for false
    specific_proc_list_input: str = "None"
    specific_proc_list = ast.literal_eval(specific_proc_list_input)
    
    
    # frapa specific
    number_hs: int = 50
    reward_limit: float = 7.5 # -11 for debug
    number_of_compress_envs = 30
    min_similarity_score: float = 0.9 # cosine sim above 0.95 seems sensible
    compress_reps: int = 5
    max_dict_in_compress: int = 1
    max_val_only: int = 1
    start_ood_level: int = 42000
    
    ## DT fracos
    finetune_fracos: bool = False
    
    #procgen specific
    easy: int = 0 # 1 = True and 0 = False and 2 = Exploration
    
    eval_interval: int = 100000
    
    #eval
    eval_specific_envs: int = 0
    eval_repeats: int = 1
    eval_batch_size: int = 50
    
    use_monochrome: int = 0 # 0 for false, 1 for true
    
    # gen_traj_specific
    method: str = "PPO"
    n_trajs: int = 100
    from_trained: bool = False
    num_repeat_seeds: int = 100
    traj_seed: int = 42
    max_action: int = 1
    
    save_network: int = 0
    load_network: int = 0
    
    if debug:
        num_envs=1
        report_epoch = num_steps*num_envs
    
    if max_eval_ep_len == 0:
        max_eval_ep_len = max_ep_length
        


def gen_trajs(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.current_depth}__{args.FraCOs_bias_factor}__{datetime.now()}"
    random.seed(args.traj_seed)
    
    if args.method == "tabularQ":
        task_seeds = [random.randint(0, 10000) for _ in range(args.n_trajs)]
    else:
        num_its = int(args.n_trajs/args.num_repeat_seeds)
        task_seeds = [random.randint(0, 10000) for _ in range(num_its)]
    
    trajectories = []
    rewards = []
    
    count = 0
    
    # random.seed(task_seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.method == "tabularQ":
        # print(args.toggle_reserve_set)
        # print(args.reserve_set)
        agent = fracos_tabularQ.fracos_tabularQ(args, track_evals=False) # by default the tabular methods will fix the mdp

        envs = SyncVectorEnv(
            [fracos_tabularQ.make_env(args.env_id, i,
                                      args.capture_video, run_name, 
                                      args.max_ep_length, args)\
             for i in range(1)],
        )
            
        
        
        next_obs, _ = envs.reset(seed=args.seed)
        
        if args.debug:
            plt.imshow(envs.envs[0].env_master.domain)
            plt.show()
        
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        
        traj = []
        c_rew = 0
        
        for step in range(args.max_ep_length):
            with torch.no_grad():
                action, agent.Q, max_qs = agent.get_action(next_obs, envs, fracos_bias_factor=args.FraCOs_bias_factor, evalu=True)
                
                next_obs_cpu = next_obs.cpu()
                traj.append([next_obs_cpu.numpy()[0], agent.cyphers[args.current_depth][action.item()]])
                
                next_obs, reward, terminations, truncations, infos, total_steps_taken = envs.fracos_step_async(action.cpu().numpy(), next_obs, next_obs, agent)
                
                if args.debug:
                    plt.imshow(envs.envs[0].env_master.domain)
                    plt.show()
                
                c_rew += reward
                
                next_done = np.logical_or(terminations, truncations)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                
                
                
            if next_done:
                break
            
        print(f"trajectory number: {count}    reward: {c_rew}")
        count += 1
        trajectories.append(traj)
        rewards.append(c_rew)
        
    elif args.method == "PPO":
        envs = SyncVectorEnv(
            [fracos_ppo.make_env(args.env_id, i, args.capture_video, run_name, args, sl=args.proc_start, nl=args.proc_num_levels, easy=args.easy) for i in range(args.num_envs)],
        )
        
        # this will do the learning
        if not args.from_trained:
            print(args.specific_proc_list)
            agent = fracos_ppo.fracos_ppo(args)
            os.makedirs(f"trained_agents/{args.env_id}", exist_ok=True)
            
            if "procgen" in args.env_id:
                h, w, c = envs.single_observation_space.shape
                model_args = [args.env_id, agent.total_action_dims]
            else:
                model_args = [envs.single_observation_space.shape, agent.total_action_dims]

            agent.ProcGenModel.save_model(f"trained_agents/{args.env_id}/checkpoint_depth_{args.current_depth}.pth")
            
            pickle.dump(model_args, open(f"trained_agents/{args.env_id}/model_args.pkl", "wb"))
            
            os.makedirs(f"fracos_clusters/{args.env_id}/a/{args.current_depth}", exist_ok=True)
            
            agent.ProcGenModel.save_model(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/policy.pth")
            
            pickle.dump(model_args, open(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/policy_args.pkl", "wb"))
                
        else:
            envs = SyncVectorEnv(
                [fracos_ppo.make_env(args.env_id, i, args.capture_video, run_name, args, sl=args.proc_start, nl=args.proc_num_levels, easy=args.easy) for i in range(args.num_envs)],
            )                

            agent = FraCOsPPOAgent(envs, args=args, top_only=args.top_only).to(device)
            
            agent.ProcGenModel.load_model(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/policy.pth")
            
            agent.network = agent.ProcGenModel.network
            agent.actor = agent.ProcGenModel.actor
            agent.critic = agent.ProcGenModel.critic
        
        
        # Loading
        # checkpoint = torch.load("checkpoint.pth")
        # network.load_state_dict(checkpoint['model_state_dict'])
        # actor.load_state_dict(checkpoint['actor_state_dict'])
        # critic.load_state_dict(checkpoint['critic_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        
        with torch.inference_mode():
            print("Getting some Ts")
            for it in range(args.n_trajs):
                # this will give us the final eval trajectory
                if args.proc_sequential:
                    sl_in = args.proc_start
                    nl_in = 1
                    
                elif args.specific_proc_list:
                    sl_in = random.choice(args.specific_proc_list)
                    print(sl_in)
                    nl_in = 1
                else:
                    try:
                        sl_in = random.choice(range(args.proc_start, args.proc_start+args.proc_num_levels))
                        nl_in = args.proc_num_levels
                    except:
                        sl_in = args.proc_start
                        nl_in = args.proc_num_levels
                        
                envs = SyncVectorEnv(
                    [fracos_ppo.make_env(args.env_id, i, args.capture_video, run_name, args, sl=sl_in, nl=nl_in) for i in range(1)],
                )
                
                if args.fix_mdp:
                    next_obs, _ = envs.reset(seed=args.seed)
                else:
                    next_obs, _ = envs.reset(seed=it) # random
                
                if args.debug:
                    plt.imshow(envs.envs[0].env_master.domain)
                    plt.show()
                    
                next_obs_np_flat = next_obs.reshape(1, -1)
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.zeros(args.num_envs).to(device)
                
                traj = []
                c_rew = 0
                
                for step in range(args.max_ep_length):
                    if args.max_action == True:
                        action, _, _, _ = agent.get_action_and_value(next_obs, next_obs_np_flat, max_dict=False)
                    else:
                        action, _, _, _ = agent.get_action_and_value(next_obs, next_obs_np_flat)
                    action = torch.tensor([action.item()], device='cuda:0')
                    next_obs_cpu = next_obs.cpu()
                    traj.append([next_obs_cpu.numpy()[0], [action.item()]])
                    
                    next_obs, reward, terminations, truncations, infos, total_steps_taken = envs.fracos_step_async(action.cpu().numpy(), next_obs, next_obs_np_flat, agent)
                    
                    c_rew += reward
                    
                    next_done = np.logical_or(terminations, truncations)
                    next_obs_np_flat = next_obs.reshape(1, -1)
                    next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                        
                        
                    if next_done:
                        break
                    
                print(f"trajectory number: {count}    reward: {c_rew}")
                count += 1
                trajectories.append(traj)
                rewards.append(c_rew)
                
                del next_obs, next_done, reward, action
                torch.cuda.empty_cache()
            
            
    if ("MetaGrid" in args.env_id) and (args.style != "grid"):
        directory = f"trajectories/e2e_traj/{args.env_id}/{args.style}"
    else:
        directory = f"trajectories/e2e_traj/{args.env_id}"
        
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        file_list = os.listdir(directory)
        for file_name in file_list:
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    file_path = directory + "/trajs.p"
    with open(file_path, 'wb') as file:
        # Serialize and save the list to the file
        pickle.dump(trajectories, file)
        
    file_path = directory + "/rews.p"
    with open(file_path, 'wb') as file:
        # Serialize and save the list to the file
        pickle.dump(rewards, file)
        
    if args.finetune_fracos: # if we have allowed them to be updated then we should now save them.
        for dt in range(args.current_depth):
            model_save_path = f"fracos_clusters/{args.env_id}/d/{dt}/transformerModel.pth"
            torch.save(agent.transformer_policies[dt].state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
        
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.specific_proc_list = ast.literal_eval(args.specific_proc_list_input)
    print(args.specific_proc_list)
    gen_trajs(args)

