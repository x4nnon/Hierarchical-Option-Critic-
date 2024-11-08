import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) ## something wrong with VSCode cwd, this fixes

# Add relevant paths
sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno/Documents/PhD/FRACOs_a")
sys.path.append("/home/x4nno_desktop/Documents/MetaGridEnv/MetaGridEnv")
sys.path.append("/home/x4nno_desktop/Documents/FRACOs_a")

sys.path.append("/app")

from gym import Wrapper
import gym as gym_old # for procgen 
import ast
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from gym.envs.registration import register 
from procgen import ProcgenEnv
from PIL import Image
from functools import reduce
from utils.sync_vector_env import SyncVectorEnv
from torch.distributions import Categorical
import torch.nn.functional as F

from matplotlib import pyplot as plt

# Register the MetaGridEnv
register( id="MetaGridEnv/metagrid-v0",
          entry_point="metagrid_gymnasium_wrapper:MetaGridEnv")

# Import your Option-Critic Agent
from OC_agents.HOC_PPO_agent import HOCAgent  # Assuming you have saved the OC agent code in option_critic.py
from utils.compatibility import EnvCompatibility

# Needed for atari
from stable_baselines3.common.atari_wrappers import (  
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['WANDB_SILENT'] = 'true'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Visualization utility
def vis_env_master(envs):
    plt.imshow(envs.envs[0].env_master.domain)

# Argument data class
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 0
    torch_deterministic: bool = False
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "HOC_StarPilot_A_QuickTest"
    wandb_entity: str = "tpcannon"
    capture_video: bool = False
    env_id: str = "procgen-starpilot"
    total_timesteps: int = 10000000
    learning_rate: float = 5e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.999
    num_minibatches: int = 4
    update_epochs: int = 2
    report_epoch: int = 81920
    anneal_ent: bool = True
    ent_coef_action: float = 0.01
    ent_coef_option: float = 0.01
    clip_coef: float = 0.1
    clip_vloss: bool = False
    vf_coef: float = 0.5
    norm_adv: bool = True #always true
    max_grad_norm: float = 0.1
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
    max_clusters_per_clusterer: int = 20
    current_depth: int = 100 # 100 for OC just to distinguish in wandb
    chain_length: int = 3
    NN_cluster_search: bool = True
    gen_strength: float = 0.33
    max_ep_length: int = 990
    fix_mdp: bool = False
    gen_traj: bool = False
    top_only: bool = False
    debug: bool = False
    proc_start: int = 1
    start_ood_level: int = 420
    proc_num_levels: int = 32
    proc_sequential: bool = False
    max_eval_ep_len: int = 1001
    sep_evals: int = 0
    specific_proc_list_input: str = "(1,2,5,6,7,9,11,12,15,16)"
    specific_proc_list = ast.literal_eval(specific_proc_list_input)
    easy: int = 1
    eval_repeats: int = 1
    use_monochrome: int = 0
    eval_interval: int = 100000
    eval_specific_envs: int = 32
    eval_batch_size: int = 32
    gae_lambda: float = 0.95
    warmup: int = 1
    debug: int = 0
    num_options: int = 25

# Plotting utilities
def plot_all_procgen_obs(next_obs, envs, option, action):
    try:
        im_d = next_obs.to("cpu")
    except:
        im_d = next_obs
    im_d_orig = np.array(im_d)
    for i in range(len(envs.envs)):
        im_d = im_d_orig[i]
        im_d = im_d / 255.0
        plt.imshow(im_d)
        plt.axis("off")
        plt.title(f"{option[i]}_{action[i]}")
        plt.show()

def plot_specific_procgen_obs(next_obs, envs, i):
    try:
        im_d = next_obs.to("cpu")
    except:
        im_d = next_obs
    im_d_orig = np.array(im_d)
    im_d = im_d_orig[i]
    im_d = im_d / 255.0
    plt.imshow(im_d)
    plt.axis("off")
    plt.show()

# Conduct Evaluations
def conduct_evals(agent, writer, global_step_truth, run_name, device):
    
    # iid
    first_ep_rewards = np.full(args.eval_specific_envs, None, dtype=object)
    first_ep_success = np.full(args.eval_specific_envs, None, dtype=object)
    cum_first_ep_rewards = np.zeros(args.eval_specific_envs)
    rep_summer = 0
    success_summer = 0
    
    current_options = torch.full((args.eval_batch_size,), -1, dtype=torch.long, device=device)

    for rep in range(args.eval_repeats):
        with torch.no_grad():
            sl_counter = args.proc_start
            for i in range(0, args.eval_specific_envs, args.eval_batch_size):
                print(sl_counter, "starting")
                    
                sls = [sl_counter + i for i in range(args.eval_batch_size)]
                print(sls)
                test_envs = SyncVectorEnv([make_env(args.env_id, sl, args.capture_video, run_name, args, sl=sl, nl=1) for sl in sls])
                test_next_obs, _ = test_envs.reset()
                test_next_obs = torch.Tensor(test_next_obs).to(device)
                

                for ts in range(args.max_eval_ep_len + 1):
                    if not torch.is_tensor(test_next_obs):
                        test_next_obs = torch.tensor(test_next_obs, dtype=torch.float32)
                    if test_next_obs.shape[-1] == 3:
                        test_next_obs = test_next_obs.permute(0, 3, 1, 2)
                    state_rep = agent.state_representation(test_next_obs/255)
                    # state_rep_o = agent.state_representation_o(test_next_obs/255)
                    
                    needs_new_option = current_options == -1  # Determine which environments need a new option
                    
                    new_options, new_option_logits, _ = agent.select_option(state_rep)
                    current_options[needs_new_option] = new_options[needs_new_option]  # Update current options
                    
                    
                    current_actions = torch.full((args.eval_batch_size,), -1, dtype=torch.long).to(device)  # Placeholder for actions
            
                    # Select actions using the intra-option policy for all environments (since all have active options now)
                    current_actions, current_a_logits, _ = agent.select_action(state_rep, current_options)
                    
                    test_next_obs, test_reward, test_terminations, test_truncations, test_infos = test_envs.step(current_actions.cpu().numpy())  # Actions are now valid for all envs
                    test_next_done = np.logical_or(test_terminations, test_truncations)
                
                    test_next_obs, test_next_done = torch.Tensor(test_next_obs).to(device), torch.Tensor(test_next_done).to(device)
                    
                
                    # Option Termination Check
                    with torch.no_grad():
                        temp_obs = test_next_obs
                        if not torch.is_tensor(temp_obs):
                            temp_obs = torch.tensor(temp_obs, dtype=torch.float32)
                        if temp_obs.shape[-1] == 3:
                            temp_obs = temp_obs.permute(0, 3, 1, 2)
                        
                            
                        temp_state_rep = agent.state_representation(temp_obs/255)
                        
                        # Only compute termination probabilities for environments where options are active
                        termination_probs = agent.termination_function(temp_state_rep, current_options)
                        terminated = torch.bernoulli(termination_probs).bool().flatten()  # Sample termination decisions
            
                        # Set current_options to -1 for environments where the option terminated
                        current_options = torch.where(terminated, torch.tensor(-1, device=device, dtype=torch.long), current_options)      
                    
                    ################
                    

                    for ve in range(len(test_reward)):
                        cum_first_ep_rewards[ve + (sl_counter - args.proc_start)] += test_reward[ve]
                        if test_terminations[ve] or test_truncations[ve]:
                            if first_ep_rewards[ve + (sl_counter - args.proc_start)] is None:
                                first_ep_success[ve + (sl_counter - args.proc_start)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                first_ep_rewards[ve + (sl_counter - args.proc_start)] = test_infos["final_info"][ve]["episode"]["r"].item()
                    if all(val is not None for val in first_ep_rewards):
                        break
                sl_counter += args.eval_batch_size
            print(first_ep_rewards)
            first_ep_rewards = np.where(first_ep_rewards == None, 0, first_ep_rewards)
            rep_summer += sum(first_ep_rewards)
            # success_summmer += sum(first_ep_success)

    writer.add_scalar("charts/avg_IID_eval_ep_rewards", rep_summer / (len(first_ep_rewards) * args.eval_repeats), global_step_truth)
    #writer.add_scalar("charts/IID_success_percentage", (success_summer * 10) / (len(first_ep_success) * args.eval_repeats), global_step_truth)
    del test_envs
    
    # OOd
    first_ep_rewards = np.full(args.eval_specific_envs, None, dtype=object)
    first_ep_success = np.full(args.eval_specific_envs, None, dtype=object)
    cum_first_ep_rewards = np.zeros(args.eval_specific_envs)
    rep_summer = 0
    success_summer = 0
    
    current_options = torch.full((args.eval_batch_size,), -1, dtype=torch.long, device=device)

    for rep in range(args.eval_repeats):
        with torch.no_grad():
            sl_counter = args.start_ood_level
            for i in range(0, args.eval_specific_envs, args.eval_batch_size):
                
                print(sl_counter, "starting")
                    
                sls = [sl_counter + i for i in range(args.eval_batch_size)]
                print(sls)
                
                test_envs = SyncVectorEnv([make_env(args.env_id, sl, args.capture_video, run_name, args, sl=sl, nl=1) for sl in sls])
                test_next_obs, _ = test_envs.reset()
                test_next_obs = torch.Tensor(test_next_obs).to(device)                   
                

                for ts in range(args.max_eval_ep_len + 1):
                    if not torch.is_tensor(test_next_obs):
                        test_next_obs = torch.tensor(test_next_obs, dtype=torch.float32)
                    if test_next_obs.shape[-1] == 3:
                        test_next_obs = test_next_obs.permute(0, 3, 1, 2)
                    state_rep = agent.state_representation(test_next_obs/255)
                    # state_rep_o = agent.state_representation_o(test_next_obs/255)
                    needs_new_option = current_options == -1  # Determine which environments need a new option
                    
                    new_options, new_option_logits, _ = agent.select_option(state_rep)
                    current_options[needs_new_option] = new_options[needs_new_option]  # Update current options
                    
                    
                    current_actions = torch.full((args.eval_batch_size,), -1, dtype=torch.long).to(device)  # Placeholder for actions
            
                    # Select actions using the intra-option policy for all environments (since all have active options now)
                    current_actions, current_a_logits, _ = agent.select_action(state_rep, current_options)
                    
                    test_next_obs, test_reward, test_terminations, test_truncations, test_infos = test_envs.step(current_actions.cpu().numpy())  # Actions are now valid for all envs
                    test_next_done = np.logical_or(test_terminations, test_truncations)
                
                    test_next_obs, test_next_done = torch.Tensor(test_next_obs).to(device), torch.Tensor(test_next_done).to(device)
                    
                
                    # Option Termination Check
                    with torch.no_grad():
                        temp_obs = test_next_obs
                        if not torch.is_tensor(temp_obs):
                            temp_obs = torch.tensor(temp_obs, dtype=torch.float32)
                        if temp_obs.shape[-1] == 3:
                            temp_obs = temp_obs.permute(0, 3, 1, 2)
                        
                            
                        temp_state_rep = agent.state_representation(temp_obs/255)
                        
                        # Only compute termination probabilities for environments where options are active
                        termination_probs = agent.termination_function(temp_state_rep, current_options)
                        terminated = torch.bernoulli(termination_probs).bool().flatten()  # Sample termination decisions
            
                        # Set current_options to -1 for environments where the option terminated
                        current_options = torch.where(terminated, torch.tensor(-1, device=device, dtype=torch.long), current_options)      
                    
                    ################
                    

                    for ve in range(len(test_reward)):
                        cum_first_ep_rewards[ve + (sl_counter - args.start_ood_level)] += test_reward[ve]
                        if test_terminations[ve] or test_truncations[ve]:
                            if first_ep_rewards[ve + (sl_counter - args.start_ood_level)] is None:
                                first_ep_success[ve + (sl_counter - args.start_ood_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                first_ep_rewards[ve + (sl_counter - args.start_ood_level)] = test_infos["final_info"][ve]["episode"]["r"].item()
                    if all(val is not None for val in first_ep_rewards):
                        break
                sl_counter += args.eval_batch_size
            print(rep_summer)
            first_ep_rewards = np.where(first_ep_rewards == None, 0, first_ep_rewards)
            rep_summer += sum(first_ep_rewards)
            # success_summer += sum(first_ep_success)

    writer.add_scalar("charts/avg_OOD_eval_ep_rewards", rep_summer / (len(first_ep_rewards) * args.eval_repeats), global_step_truth)
    #writer.add_scalar("charts/OOD_success_percentage", (success_summer * 10) / (len(first_ep_success) * args.eval_repeats), global_step_truth)
    del test_envs


def make_env(env_id, idx, capture_video, run_name, args, sl=1, nl=10, enforce_mes=False, easy=True, seed=0):
    def thunk():
        if "procgen" in args.env_id: # need to create a specific method for making these vecenvs
            # The max ep length is handled in the fracos wrapper - the envcompatibility will give warnings about 
            # early reset being ignored, but it does get truncated by the fracos_wrapper. So you can ignore these warnings.
            # print(sl_in)
            if easy:
                # print(sl_in)
                if args.use_monochrome:
                    env = gym_old.make(args.env_id, num_levels=nl, start_level=sl, distribution_mode="easy", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
                else:
                    env = gym_old.make(args.env_id, num_levels=nl, start_level=sl, distribution_mode="easy", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=False, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
            else:
                # print(sl_in)
                if args.use_monochrome:
                    env = gym_old.make(args.env_id, num_levels=nl, start_level=sl, distribution_mode="hard", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=True, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
                else:
                    env = gym_old.make(args.env_id, num_levels=nl, start_level=sl, distribution_mode="hard", use_backgrounds=False, restrict_themes=True, use_monochrome_assets=False, use_sequential_levels=args.proc_sequential, rand_seed=int(seed)) # change this will only do one env
            
            env.observation_space = gym.spaces.Box(0,255,(64,64,3), "int")
            env.action_space = gym.spaces.Discrete(env.action_space.n)
            
            # env.action_space
            #envs = gym_old.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
            
            env = EnvCompatibility(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym_old.wrappers.NormalizeReward(env, gamma=args.gamma)
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
            
        
        else:
            env = gym.make(env_id, max_episode_steps=args.max_ep_length)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        

        return env

    return thunk


def hoc(args):
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"OC_{args.env_id}__{args.exp_name}__{args.seed}__{datetime.now()}"

    if args.track and not args.debug:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args, sl=args.proc_start, nl=args.proc_num_levels, easy=args.easy) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Initialize Option-Critic agent
    agent = HOCAgent(
        input_channels=3, 
        num_actions=envs.single_action_space.n, 
        num_options=args.num_options,  # Number of options for OC agent
        hidden_dim=256,
        gamma=args.gamma,
        learning_rate=args.learning_rate
    ).to(device)
    
    
    ####### load if not in warmup 
    # !!! This needs changing once we have saved sensibly
    if not args.warmup:
        load_path = f"OC_policies/{args.env_id}"
        agent.load(load_path, exclude_meta=True)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    options_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)  # Track active options
    actions_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)  # Track primitive actions
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs_options = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step_truth = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    
    
    
    with torch.no_grad():
        temp_obs = next_obs
        if not torch.is_tensor(temp_obs):
            temp_obs = torch.tensor(temp_obs, dtype=torch.float32)
        if temp_obs.shape[-1] == 3:
            temp_obs = temp_obs.permute(0, 3, 1, 2)
        

    current_options = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
    
    
    
    epoch_res_count = 0
    global_decisions = 0
    if not args.warmup:
        conduct_evals(agent, writer, 0, run_name, device)
        epoch_res_count += 1
        
    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (global_step_truth / args.total_timesteps)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate
            
        if args.anneal_ent:
            frac = 1.0 - (global_step_truth / args.total_timesteps)
            ent_action_coef_now = frac * args.ent_coef_action
            ent_option_coef_now = frac * args.ent_coef_option
        else:
            ent_action_coef_now = args.ent_coef_action
            ent_option_coef_now = args.ent_coef_option
            

        for step in range(0, args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done
            
            if not torch.is_tensor(next_obs):
                next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if next_obs.shape[-1] == 3:
                next_obs = next_obs.permute(0, 3, 1, 2)
                
            state_rep = agent.state_representation(next_obs/255)
            # state_rep_o = agent.state_representation_o(next_obs/255)
                
            with torch.no_grad():
                # For environments where the option has terminated (or no active option), we select a new option
                needs_new_option = current_options == -1  # Determine which environments need a new option
                
                global_decisions += torch.nonzero(current_options == -1).numel()
                
                new_options, new_option_logits, _ = agent.select_option(state_rep)
                current_options[needs_new_option] = new_options[needs_new_option]  # Update current options
                
                logprobs_options[step] = new_option_logits
                
                current_actions = torch.full((args.num_envs,), -1, dtype=torch.long).to(device)  # Placeholder for actions
        
                # Select actions using the intra-option policy for all environments (since all have active options now)
                current_actions, current_a_logits, _ = agent.select_action(state_rep, current_options)
                
                logprobs_actions[step] = current_a_logits
                value = agent.compute_value(state_rep).flatten()
                values[step] = value
                
                
            
        
            # Store the active options and primitive actions in the buffer for later updates
            options_buffer[step] = current_options  # Store the option that was active during this step
            actions_buffer[step] = current_actions  # Store the primitive action taken during this step
            
        
            # Step the environment with the selected primitive actions (for all environments in batch)
            next_obs, reward, terminations, truncations, infos = envs.step(current_actions.cpu().numpy())  # Actions are now valid for all envs
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).reshape(-1)
            
            if args.debug:
                plot_all_procgen_obs(next_obs, envs, current_options, current_actions)
            
            if reward.any():
                pass # debug point
        
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            
            if "final_info" in infos:
                ref=0
                for info in infos["final_info"]:
                    if info and ("episode" in info):
                        print(f"global_step={global_step_truth}, ep_r={info['episode']['r']}, ep_l={info['episode']['l']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step_truth)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step_truth)
                        # plot_specific_procgen_obs(next_obs, envs, ref)
                    ref += 1
        
            # Option Termination Check
            with torch.no_grad():
                temp_obs = next_obs
                if not torch.is_tensor(temp_obs):
                    temp_obs = torch.tensor(temp_obs, dtype=torch.float32)
                if temp_obs.shape[-1] == 3:
                    temp_obs = temp_obs.permute(0, 3, 1, 2)
                
                    
                temp_state_rep = agent.state_representation(temp_obs/255)
                
                # Only compute termination probabilities for environments where options are active
                termination_probs = agent.termination_function(temp_state_rep, current_options)
                terminated = torch.bernoulli(termination_probs).bool().flatten()  # Sample termination decisions
    
                # Set current_options to -1 for environments where the option terminated
                current_options = torch.where(terminated, torch.tensor(-1, device=device, dtype=torch.long), current_options)        
            global_step_truth += args.num_envs
            
            
            # print(f"report at {args.report_epoch*epoch_res_count} steps")
            # print(f"current step is {global_step_truth}")
            if (global_step_truth == args.report_epoch*epoch_res_count) and (not args.warmup):
                conduct_evals(agent, writer, global_step_truth, run_name, device)
                epoch_res_count += 1
                
        ## PPO UPDATE LOGIC BELOW 
        
        b4 = time.time() # remove
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.compute_value(temp_state_rep).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                # reward adjust (should be a wrapper but results gathered using this)
                adjusted_rewards = rewards[t]
                
                delta = adjusted_rewards + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
            option_advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_a_logprobs = logprobs_actions.reshape(-1)
        b_o_logprobs = logprobs_options.reshape(-1)
        b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_options = options_buffer.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_option_advantages = option_advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        a_clipfracs = []
        o_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                update_state_rep = agent.state_representation(b_obs[mb_inds].permute(0, 3, 1, 2)/255)
                update_state_rep_o = agent.state_representation(b_obs[mb_inds].permute(0, 3, 1, 2)/255)
                
                _, actionlogprobs, actionentropy = agent.select_action(update_state_rep, b_options[mb_inds], b_actions[mb_inds])
                _, optionlogprobs, optionentropy = agent.select_option(update_state_rep_o, b_options[mb_inds])
                newvalue = agent.compute_value(update_state_rep)

                actionlogratio = actionlogprobs - b_a_logprobs[mb_inds]
                optionlogratio = optionlogprobs - b_o_logprobs[mb_inds]
                a_ratio = actionlogratio.exp()
                o_ratio = optionlogratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_a_approx_kl = (-actionlogratio).mean()
                    old_o_approx_kl = (-optionlogratio).mean()
                    approx_a_kl = ((a_ratio - 1) - actionlogratio).mean()
                    approx_o_kl = ((o_ratio - 1) - optionlogratio).mean()
                    a_clipfracs += [((a_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    o_clipfracs += [((o_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_o_advantages = b_option_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # action Policy loss
                a_pg_loss1 = -mb_advantages * a_ratio
                a_pg_loss2 = -mb_advantages * torch.clamp(a_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                a_pg_loss = torch.max(a_pg_loss1, a_pg_loss2).mean()
                
                # option Policy loss
                o_pg_loss1 = -mb_o_advantages * o_ratio
                o_pg_loss2 = -mb_o_advantages * torch.clamp(o_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                o_pg_loss = torch.max(o_pg_loss1, o_pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # termination loss

                termination_probs = agent.termination_function(update_state_rep, b_options[mb_inds])
                termination_advantage = b_returns[mb_inds] - newvalue.detach()
                termination_loss = (termination_probs * termination_advantage).mean()

                a_entropy_loss = actionentropy.mean()
                o_entropy_loss = optionentropy.mean()
                loss = a_pg_loss - ent_action_coef_now * a_entropy_loss  \
                    + o_pg_loss - ent_option_coef_now * o_entropy_loss \
                        + v_loss * args.vf_coef + termination_loss*0.2

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.all_params, args.max_grad_norm)
                optimizer.step()

            # if args.target_kl is not None and approx_kl > args.target_kl: # not implemented this
            #     break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        global_step_truth += args.num_envs

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_truth)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step_truth)
        writer.add_scalar("losses/a_policy_loss", a_pg_loss.item(), global_step_truth)
        writer.add_scalar("losses/o_policy_loss", o_pg_loss.item(), global_step_truth)
        writer.add_scalar("losses/a_entropy", a_entropy_loss.item(), global_step_truth)
        writer.add_scalar("losses/o_entropy", o_entropy_loss.item(), global_step_truth)
        writer.add_scalar("losses/old_a_approx_kl", old_a_approx_kl.item(), global_step_truth)
        writer.add_scalar("losses/old_o_approx_kl", old_o_approx_kl.item(), global_step_truth)
        writer.add_scalar("losses/approx_a_kl", approx_a_kl.item(), global_step_truth)
        writer.add_scalar("losses/approx_o_kl", approx_o_kl.item(), global_step_truth)
        writer.add_scalar("losses/a_clipfrac", np.mean(a_clipfracs), global_step_truth)
        writer.add_scalar("losses/o_clipfrac", np.mean(o_clipfracs), global_step_truth)
        writer.add_scalar("losses/explained_variance", explained_var, global_step_truth)
        writer.add_scalar("losses/termination_loss", termination_loss, global_step_truth)
        print("SPS:", int(global_step_truth / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step_truth / (time.time() - start_time)), global_step_truth)
        writer.add_scalar("charts/decisions", int(global_decisions), global_step_truth)

        after = time.time()
        print("Time taken: ", after-b4)
        
    
    if args.warmup:
        
        folder_path = f"OC_policies/{args.env_id}"
        agent.save(folder_path)
        
        
    envs.close()
    writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.report_epoch = args.num_envs * args.num_steps * 20
    print(f"report at {args.report_epoch} steps")
    
    hoc(args)
