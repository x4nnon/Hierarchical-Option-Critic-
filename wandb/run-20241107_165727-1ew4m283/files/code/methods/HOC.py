import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) ## something wrong with VSCode cwd, this fixes

# Add relevant paths
# sys.path.append("/home/x4nno/Documents/PhD/MetaGridEnv/MetaGridEnv")
# sys.path.append("/home/x4nno/Documents/PhD/FRACOs_a")
# sys.path.append("/home/x4nno_desktop/Documents/MetaGridEnv/MetaGridEnv")
# sys.path.append("/home/x4nno_desktop/Documents/FRACOs_a")

# sys.path.append("/app")

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


# Import your Option-Critic Agent
from OC_agents.HOC_agent import HOCAgent  # Assuming you have saved the OC agent code in option_critic.py
from utils.compatibility import EnvCompatibility

# Needed for atari
from stable_baselines3.common.atari_wrappers import (  
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

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
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_envs: int = 8
    num_steps: int = 256
    anneal_lr: bool = True
    gamma: float = 0.999
    num_minibatches: int = 1
    update_epochs: int = 4
    report_epoch: int = 81920
    anneal_ent: bool = True
    ent_coef_action: float = 0.005
    ent_coef_option: float = 0.05
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
    eval_specific_envs: int = proc_num_levels
    eval_batch_size: int = eval_specific_envs // 2
    gae_lambda: float = 0.95
    warmup: int = 1
    debug: int = 0

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
    first_ep_rewards = np.full(args.eval_specific_envs, None, dtype=object)
    first_ep_success = np.full(args.eval_specific_envs, None, dtype=object)
    cum_first_ep_rewards = np.zeros(args.eval_specific_envs)
    rep_summer = 0
    success_summer = 0

    for rep in range(args.eval_repeats):
        with torch.no_grad():
            sl_counter = args.proc_start
            for i in range(0, args.eval_specific_envs, args.eval_batch_size):
                sls = [sl_counter + i for i in range(args.eval_batch_size)]
                test_envs = SyncVectorEnv([make_env(args.env_id, sl, args.capture_video, run_name, args, sl=sl, nl=1) for sl in sls])
                test_next_obs, _ = test_envs.reset()
                test_next_obs = torch.Tensor(test_next_obs).to(device)
                test_next_obs_np_flat = test_next_obs.reshape(args.eval_batch_size, -1)

                for ts in range(args.max_eval_ep_len + 1):
                    option, action = agent.select_option_or_action(test_next_obs)
                    if action is None:
                        action = agent.select_action(test_next_obs, option)

                    test_next_obs, test_reward, test_terminations, test_truncations, test_infos, _ = test_envs.step(
                        action.cpu().numpy()
                    )
                    test_next_obs = torch.Tensor(test_next_obs).to(device)

                    for ve in range(len(test_reward)):
                        cum_first_ep_rewards[ve + (sl_counter - args.proc_start)] += test_reward[ve]
                        if test_terminations[ve] or test_truncations[ve]:
                            if first_ep_rewards[ve + (sl_counter - args.proc_start)] is None:
                                first_ep_success[ve + (sl_counter - args.proc_start)] = test_infos["final_info"][ve]["episode"]["r"].item()
                                first_ep_rewards[ve + (sl_counter - args.proc_start)] = test_infos["final_info"][ve]["episode"]["r"].item()
                    if all(val is not None for val in first_ep_rewards):
                        break
                sl_counter += args.eval_batch_size
            rep_summer += sum(first_ep_rewards)
            success_summer += sum(first_ep_success)

    writer.add_scalar("charts/avg_IID_eval_ep_rewards", rep_summer / (len(first_ep_rewards) * args.eval_repeats), global_step_truth)
    writer.add_scalar("charts/IID_success_percentage", (success_summer * 10) / (len(first_ep_success) * args.eval_repeats), global_step_truth)
    del test_envs

# Make environment function
def make_env(env_id, idx, capture_video, run_name, args, sl=1, nl=10, enforce_mes=False, easy=True, seed=0):
    def thunk():
        sl_in = random.choice(args.specific_proc_list) if args.specific_proc_list else sl
        nl_in = 1 if args.specific_proc_list else nl
        if "procgen" in args.env_id:
            if easy:
                env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="easy", use_backgrounds=False, rand_seed=int(seed))
            else:
                env = gym_old.make(args.env_id, num_levels=nl_in, start_level=sl_in, distribution_mode="hard", use_backgrounds=False, rand_seed=int(seed))
            env.observation_space = gym.spaces.Box(0,255,(64,64,3), "int")
            env.action_space = gym.spaces.Discrete(env.action_space.n)
            
            # env.action_space
            #envs = gym_old.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
            
            env = EnvCompatibility(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym_old.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym_old.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else:
            env = gym.make(env_id, max_episode_steps=args.max_ep_length)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def compute_returns_and_advantages(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    # Initialize placeholders for returns and advantages
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    
    last_gae_lambda = 0
    for t in reversed(range(rewards.size(0))):
        if t == rewards.size(0) - 1:
            next_non_terminal = 1.0 - dones[-1]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[t] = last_gae_lambda = delta + gamma * lam * next_non_terminal * last_gae_lambda
    
    returns = advantages + values
    return returns, advantages


def oc(args):
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
        num_options=4,  # Number of options for OC agent
        hidden_dim=256,
        gamma=args.gamma,
        learning_rate=args.learning_rate
    ).to(device)
    
    
    ####### load if not in warmup
    if not args.warmup:
        state_rep_dict = torch.load(f"OC_policies/{args.env_id}/termination.pth")
        agent.conv1.load_state_dict(state_rep_dict['conv1'])
        agent.conv2.load_state_dict(state_rep_dict['conv2'])
        agent.conv3.load_state_dict(state_rep_dict['conv3'])
        agent.fc1.load_state_dict(state_rep_dict['fc1'])
        agent.termination.load_state_dict(state_rep_dict['termination'])
        
        agent.intra_option_policy.load_state_dict(torch.load(f"OC_policies/{args.env_id}/intra.pth"))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    options_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)  # Track active options
    actions_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)  # Track primitive actions
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step_truth = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    current_options = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
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
        
            with torch.no_grad():
                # For environments where the option has terminated (or no active option), we select a new option
                needs_new_option = current_options == -1  # Determine which environments need a new option
                
                if needs_new_option.any():
                    # Select new options for environments where needed
                    new_options = agent.select_option(next_obs[needs_new_option])
                    current_options[needs_new_option] = new_options  # Update current options
        
                option_mask = current_options != -1  # Mask for environments where an option is active
                current_actions = torch.full((args.num_envs,), -1, dtype=torch.long).to(device)  # Placeholder for actions
        
                # Select actions using the intra-option policy for all environments (since all have active options now)
                intra_option_actions = agent.select_action(next_obs, current_options)
                current_actions = intra_option_actions  # Assign the actions for all environments
                
                _, value = agent.compute_q_value(next_obs, current_options, current_actions)
                values[step] = value
        
            # Compute the Q-value (value) for the current state, for all environments
            
        
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
                option_mask = current_options != -1  # Mask for environments where an option is active
                if option_mask.any():
                    # Only compute termination probabilities for environments where options are active
                    termination_probs = agent.termination_function(next_obs[option_mask], current_options[option_mask])
                    terminated = torch.bernoulli(termination_probs).bool()  # Sample termination decisions
        
                    # Set current_options to -1 for environments where the option terminated
                    old_current_options = current_options.clone()
                    current_options[option_mask] = torch.where(terminated, torch.tensor(-1, device=device, dtype=torch.long), current_options[option_mask])        
            global_step_truth += args.num_envs

        for epoch in range(args.update_epochs):
    
            # Compute the next value for the final step
            with torch.no_grad():
                _, next_value = agent.compute_q_value(next_obs, old_current_options, current_actions)
        
            # Initialize advantage buffers
            option_advantages = torch.zeros_like(rewards).to(device)  # For policy over options
            action_advantages = torch.zeros_like(rewards).to(device)  # For intra-option policies
        
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
        
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
        
                # Delta for GAE (action advantages)
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                action_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        
                # Compute option advantages

                q_values_for_options = agent.compute_q_values_for_all_options(obs[t])  # Get Q-values for all options at time t
                q_value_for_past_option = torch.gather(q_values_for_options, 1, options_buffer[t].reshape(-1, 1)).squeeze(1)
    
                if obs[t].shape[-1] == 3:
                    state = obs[t].permute(0, 3, 1, 2) 
    
                option_logits = agent.policy_over_options(state / 255)  # Get option logits at time t
                option_probs = Categorical(logits=option_logits)  # Softmax to convert logits to probabilities
                V_state = torch.sum(option_probs.probs * q_values_for_options, dim=-1)  # V(s), the value function over all options
    
                # Option advantage: A(s, o) = Q(s, o) - V(s)
                option_advantages[t] = q_value_for_past_option - V_state  # Use the value for the active option
        
            # Compute action returns using the action advantages
            returns = action_advantages + values  # Action returns based on action advantages
        
            # Flatten the batch for updating the policies
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_actions = actions_buffer.reshape((-1,) + envs.single_action_space.shape)
            b_options = options_buffer.reshape((-1,))  # Flattened options
            b_action_advantages = action_advantages.reshape(-1)
            b_option_advantages = option_advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
        
            # ------------------- Perform updates -------------------
            
            _, newvalues = agent.compute_q_value(b_obs, b_options, b_actions)

            agent.update_combined_all(b_obs, b_options, b_actions, b_option_advantages,
                                      b_action_advantages, newvalues, b_returns, global_step_truth,
                                      writer, args, ent_action_coef_now, ent_option_coef_now)
            
        print("update complete")
        
        torch.cuda.empty_cache()
        if global_step_truth >= args.total_timesteps:
            if not args.warmup:
                conduct_evals(agent, writer, global_step_truth, run_name, device)
                    
    
    if args.warmup:
        
        # check make folder
        folder_path = f"OC_policies/{args.env_id}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")
        else:
            print(f"Folder '{folder_path}' already exists.")
        
        # save the intra_option
        torch.save(agent.intra_option_policy.state_dict(), f"OC_policies/{args.env_id}/intra.pth")
        
        # save the state_rep and termination:
        state_rep_dict = {
            'conv1': agent.conv1.state_dict(),
            'conv2': agent.conv2.state_dict(),
            'conv3': agent.conv3.state_dict(),
            'fc1': agent.fc1.state_dict(),
            'termination': agent.termination.state_dict(),
        }
        torch.save(state_rep_dict, f"OC_policies/{args.env_id}/termination.pth")
        
        
    envs.close()
    writer.close()


def main_training_loop(agent, args, writer, envs, device):
    global_step_truth = 0
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    options_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    meta_options_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    actions_buffer = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # For action values
    option_values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # For option values
    meta_option_values = torch.zeros((args.num_steps, args.num_envs)).to(device)  # For meta-option values

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    current_options = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
    current_meta_options = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)

    for iteration in range(1, args.num_iterations + 1):
        for step in range(0, args.num_steps):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                if (current_meta_options == -1).any():
                    new_meta_options = agent.select_meta_option(next_obs[current_meta_options == -1])
                    current_meta_options[current_meta_options == -1] = new_meta_options

                if (current_options == -1).any():
                    new_options = agent.select_option(next_obs[current_options == -1], current_meta_options[current_options == -1])
                    current_options[current_options == -1] = new_options

                actions = agent.select_action(next_obs, current_options)
                action_value, option_value, meta_option_value = agent.compute_q_values(next_obs, actions, current_options, current_meta_options)

            options_buffer[step] = current_options
            meta_options_buffer[step] = current_meta_options
            actions_buffer[step] = actions
            
            values[step] = action_value
            option_values[step] = option_value
            meta_option_values[step] = meta_option_value

            next_obs, reward, terminations, truncations, infos = envs.step(actions.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).reshape(-1)
            next_done = np.logical_or(terminations, truncations)

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

            global_step_truth += args.num_envs

            # Check for option termination
            with torch.no_grad():
                option_termination_probs = agent.termination_function_option(next_obs, current_options)
                option_terminated = torch.bernoulli(option_termination_probs).bool()
                current_options[option_terminated] = -1

                # Check for meta-option termination
                meta_option_termination_probs = agent.termination_function_meta(next_obs, current_meta_options)
                meta_option_terminated = torch.bernoulli(meta_option_termination_probs).bool()
                current_meta_options[meta_option_terminated] = -1

        batch = (obs, actions_buffer, options_buffer, meta_options_buffer, rewards, dones, values, option_values, meta_option_values)

        total_loss = agent.update_function(batch, args, writer, global_step_truth)
        print(f"Total loss: {total_loss}")

if __name__ == "__main__":
    args = tyro.cli(Args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"HOC_{args.env_id}__{args.seed}__{datetime.now()}"

    envs = SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name, args, sl=args.proc_start, nl=args.proc_num_levels, easy=args.easy) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    if args.track and not args.debug:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=run_name, monitor_gym=True, save_code=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    agent = HOCAgent(
            input_channels=3,
            num_meta_options=4,
            num_options=16,
            num_actions=envs.single_action_space.n, 
            hidden_dim=256,
            gamma=args.gamma,
            learning_rate=args.learning_rate
        ).to(device)

    main_training_loop(agent, args, writer, envs, device)
