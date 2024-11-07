#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 09:58:01 2024

@author: x4nno
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProcGenMetaActor(nn.Module):
    def __init__(self, num_meta_options, hidden_dim):
        super(ProcGenMetaActor, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.metafc = nn.Linear(hidden_dim, num_meta_options)

    def forward(self, state_rep):
        x = F.relu(self.fc1(state_rep))
        meta_option_logits = self.metafc(x)
        return meta_option_logits


class ProcGenOptionActor(nn.Module):
    def __init__(self, num_options, hidden_dim):
        super(ProcGenOptionActor, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.optionfc = nn.Linear(hidden_dim, num_options)

    def forward(self, state_rep):
        x = F.relu(self.fc1(state_rep))
        option_logits = self.optionfc(x)
        return option_logits


class ProcGenActor(nn.Module):
    def __init__(self, num_actions, hidden_dim):
        super(ProcGenActor, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.actions = nn.Linear(hidden_dim, num_actions)

    def forward(self, state_rep):
        x = F.relu(self.fc1(state_rep))
        action_logits = self.actions(x)
        return action_logits


class ProcGenCritic(nn.Module):
    def __init__(self, num_actions, hidden_dim):
        super(ProcGenCritic, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        # Output Q-values for both options and actions
        self.fc_actions = nn.Linear(hidden_dim, num_actions)  # Q-values for actions

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Final hidden representation of the state
        # Compute Q-values for both options and primitive actions
        q_actions = self.fc_actions(x)  # Shape: (batch_size, num_actions)
        return q_actions  # Return Q-values for options and actions
    
class ProcGenOptionCritic(nn.Module):
    def __init__(self, num_options, hidden_dim):
        super(ProcGenOptionCritic, self).__init__()
        self.num_options = num_options

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        
        # Output Q-values for both options and actions
        self.fc_options = nn.Linear(hidden_dim, num_options)  # Q-values for options
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Final hidden representation of the state
        # Compute Q-values for both options and primitive actions
        q_options = self.fc_options(x)  # Shape: (batch_size, num_options)
        return q_options  # Return Q-values for options
    

class ProcGenMetaCritic(nn.Module):
    def __init__(self, num_meta_options, hidden_dim):
        super(ProcGenMetaCritic, self).__init__()
        self.num_meta_options = num_meta_options

        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        
        # Output Q-values for both options and actions
        self.fc_meta_options = nn.Linear(hidden_dim, num_meta_options)  # Q-values for meta-options
        
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Final hidden representation of the state
        # Compute Q-values for both options and primitive actions
        q_meta_options = self.fc_meta_options(x)  # Shape: (batch_size, num_meta_options)
        return q_meta_options  # Return Q-values for meta-options
    

class SharedStateRepresentation(nn.Module):
    def __init__(self, input_channels):
        super(SharedStateRepresentation, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return x
    
class OptionTermination(nn.Module):
    def __init__(self, hidden_dim):
        super(OptionTermination, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.T = nn.Linear(hidden_dim, 1)

    def forward(self, state_rep):
        x = F.relu(self.fc1(state_rep))
        termination_prob = torch.sigmoid(self.T(x))
        return termination_prob.squeeze(-1)
    
class MetaTermination(nn.Module):
    def __init__(self, hidden_dim):
        super(MetaTermination, self).__init__()
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.T = nn.Linear(hidden_dim, 1)

    def forward(self, state_rep):
        x = F.relu(self.fc1(state_rep))
        termination_prob = torch.sigmoid(self.T(x))
        return termination_prob.squeeze(-1)
    

class HOCAgent(nn.Module):
    def __init__(self, input_channels, num_meta_options, num_options, num_actions, hidden_dim, gamma=0.99, learning_rate=0.001):
        super().__init__()
        self.num_meta_options = num_meta_options
        self.num_options = num_options
        self.num_actions = num_actions
        self.gamma = gamma

        # Meta-option policy
        self.meta_option_policy = ProcGenMetaActor(num_meta_options, hidden_dim).to(device) 
        # Policy over options
        self.policys_over_options = [ProcGenOptionActor(num_options, hidden_dim).to(device) for _ in range(num_meta_options)]
        # Intra-option policy
        self.actors_policy = [ProcGenActor(num_actions, hidden_dim).to(device) for _ in range(num_options)]
        # Critic for option-value estimation
        self.option_critic = ProcGenOptionCritic(num_options, hidden_dim).to(device)
        # Critic for meta-option-value estimation
        self.meta_critic = ProcGenMetaCritic(num_meta_options, hidden_dim).to(device)
        #critic over actions
        self.critic_over_actions = ProcGenCritic(num_actions, hidden_dim).to(device)
        #shared state representation
        self.forward_state_rep = SharedStateRepresentation(input_channels).to(device)

        # Termination networks for options and meta-options
        self.option_termination = [OptionTermination(hidden_dim).to(device) for _ in range(num_options)]
        self.meta_termination = [MetaTermination(hidden_dim).to(device) for _ in range(num_meta_options)]

        # I think ... check this in the debugger.
        all_policy_over_options_params = [list(self.policys_over_options[i].parameters()) for i in range(num_meta_options)]
        all_actors_policy_params = [list(self.actors_policy[i].parameters()) for i in range(num_options)]
        all_option_termination_params = [list(self.option_termination[i].parameters()) for i in range(num_options)]
        all_meta_termination_params = [list(self.meta_termination[i].parameters()) for i in range(num_meta_options)]

        # Collect all parameters for joint optimization
        self.all_params = list(self.meta_option_policy.parameters()) + list(self.option_critic.parameters()) \
                          + list(self.meta_critic.parameters())
        
        for param in all_policy_over_options_params:
            self.all_params.extend(param)
        for param in all_actors_policy_params:
            self.all_params.extend(param)
        for param in all_option_termination_params:
            self.all_params.extend(param)
        for param in all_meta_termination_params:
            self.all_params.extend(param)

        self.optimizer = optim.Adam(self.all_params, lr=learning_rate)
                
    def select_meta_option(self, state, meta_options=None):
        """
        Select a meta-option using the meta-option policy.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)
        meta_option_logits = self.meta_option_policy(state_rep)
        meta_option_probs = Categorical(logits=meta_option_logits)
        if meta_options is None:
            meta_option = meta_option_probs.sample()
        else:
            meta_option = meta_options
        return meta_option, meta_option_logits

    def select_option(self, state, meta_option, options=None):
        """
        Select an option based on the current meta-option.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)

        batch_size = state_rep.size(0)
        option_logits_list = torch.zeros(batch_size, self.num_options, device=state_rep.device)

        # Process all meta-options in parallel by gathering the relevant states for each meta-option
        for meta_option_idx in range(self.num_meta_options):
            mask = (meta_option == meta_option_idx)  # Mask to select states with the current meta-option
            
            if mask.sum() > 0:  # Only process if there are states with this meta-option
                selected_states = state_rep[mask]
                option_logits_list[mask] = self.policys_over_options[meta_option_idx](selected_states)
        
        option_probs = Categorical(logits=option_logits_list)
        if options is None:
            option = option_probs.sample()
        else:
            option = options
        return option, option_logits_list

    def select_action(self, state, option, actions=None):
        """
        Select an action using the intra-option policy for the given option.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)

        batch_size = state_rep.size(0)
        action_logits_list = torch.zeros(batch_size, self.num_actions, device=state_rep.device)

        # Process all options in parallel by gathering the relevant states for each option
        for option_idx in range(self.num_options):
            mask = (option == option_idx)  # Mask to select states with the current option
            
            if mask.sum() > 0:  # Only process if there are states with this option
                selected_states = state_rep[mask]
                action_logits_list[mask] = self.actors_policy[option_idx](selected_states)
        
        action_probs = Categorical(logits=action_logits_list)
        if actions is None: 
            action = action_probs.sample()
        else:
            action = actions
        return action, action_logits_list

    def termination_function_meta(self, state, meta_option):
        """
        Compute termination probability for meta-options.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)
        termination_probs = torch.zeros(state_rep.size(0), device=state_rep.device)
        for i in range(self.num_meta_options):
            mask = (meta_option == i)
            if mask.sum() > 0:
                termination_prob = self.meta_termination[i](state_rep[mask])
                termination_probs[mask] = termination_prob
        return termination_probs

    def termination_function_option(self, state, option):
        """
        Compute termination probability for regular options.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)
        termination_probs = torch.zeros(state_rep.size(0), device=state_rep.device)
        for i in range(self.num_options):
            mask = (option == i)
            if mask.sum() > 0:
                termination_prob = self.option_termination[i](state_rep[mask])
                termination_probs[mask] = termination_prob
        return termination_probs

    def compute_q_values(self, state, action, option, meta_option):
        """
        Compute the Q-values for the given state, option, and meta-option.
        """
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)
        state_rep = self.forward_state_rep(state)

        # Compute Q-value for meta-options
        q_meta_options = self.meta_critic(state_rep)
        q_meta_option_value = torch.gather(q_meta_options, 1, meta_option.unsqueeze(1)).squeeze(1)
        
        # compute Q-value for actions
        q_actions = self.critic_over_actions(state_rep)
        q_action_value = torch.gather(q_actions, 1, action.unsqueeze(1)).squeeze(1)
    
        # Compute Q-value for options
        q_options = self.option_critic(state_rep)
        q_option_value = torch.gather(q_options, 1, option.unsqueeze(1)).squeeze(1)

        return q_action_value, q_option_value, q_meta_option_value
    

    def update_function(self, batch, args, writer, global_step_truth, envs):
        states, actions, options, meta_options, rewards, dones, values, option_values, meta_option_values = batch

        # flatten the batch
        b_states = states.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_options = options.reshape((-1,) + envs.single_action_space.shape)
        b_meta_options = meta_options.reshape((-1,) + envs.single_action_space.shape)
        b_rewards = rewards.reshape(-1)
        b_option_values = option_values.reshape(-1)
        b_meta_option_values = meta_option_values.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        if b_states.shape[-1] == 3:
            b_states = b_states.permute(0, 3, 1, 2)
        # Compute the current state representation
        # state_reps = self.forward_state_rep(b_states)

        # Use pre-computed Q-values from the buffer
        q_action_values = b_values
        q_option_values = b_option_values
        q_meta_values = b_meta_option_values

        # Compute returns and advantages using GAE
        meta_advantages, meta_returns = self.compute_advantages(
            b_rewards, q_meta_values, b_dones, args.gamma, args.gae_lambda
        )
        option_advantages, option_returns = self.compute_advantages(
            b_rewards, q_option_values, b_dones, args.gamma, args.gae_lambda
        )
        action_advantages, action_returns = self.compute_advantages(
            b_rewards, q_action_values, b_dones, args.gamma, args.gae_lambda
        )

        # Compute policy losses for meta-options
        _, meta_option_logits = self.select_meta_option(b_states, meta_options=b_meta_options)
        meta_option_probs = Categorical(logits=meta_option_logits)
        meta_log_probs = meta_option_probs.log_prob(b_meta_options) 
        meta_policy_loss = -(meta_log_probs * meta_advantages).mean()
        meta_entropy = meta_option_probs.entropy().mean()
        meta_policy_loss -= args.meta_ent_coef * meta_entropy

        # Compute policy losses for options
        _, option_logits = self.select_option(b_states, b_meta_options, options=b_options)
        option_probs = Categorical(logits=option_logits)
        option_log_probs = option_probs.log_prob(b_options)
        option_policy_loss = -(option_log_probs * option_advantages).mean()
        option_entropy = option_probs.entropy().mean()
        option_policy_loss -= args.option_ent_coef * option_entropy

        # Compute action policy losses
        _, action_logits = self.select_action(b_states, b_options, actions=b_actions)
        action_probs = Categorical(logits=action_logits)
        action_log_probs = action_probs.log_prob(b_actions)
        action_policy_loss = -(action_log_probs * action_advantages).mean()
        action_entropy = action_probs.entropy().mean()
        action_policy_loss -= args.action_ent_coef * action_entropy

        # Compute termination losses
        termination_probs_meta = self.termination_function_meta(b_states, b_meta_options)
        termination_loss_meta = -(meta_advantages * (1 - termination_probs_meta)).mean()

        termination_probs_option = self.termination_function_option(b_states, b_options)
        termination_loss_option = -(option_advantages * (1 - termination_probs_option)).mean()

        # Compute critic losses
        critic_loss_meta = 0.5 * ((q_meta_values - meta_returns) ** 2).mean()
        critic_loss_option = 0.5 * ((q_option_values - option_returns) ** 2).mean()
        critic_loss_action = 0.5 * ((q_action_values - action_returns) ** 2).mean()

        # Total loss
        total_loss = meta_policy_loss + option_policy_loss + action_policy_loss + \
                    termination_loss_meta + termination_loss_option + \
                    critic_loss_meta + critic_loss_option + critic_loss_action

        # Logging
        writer.add_scalar("losses/total_loss", total_loss, global_step_truth)
        writer.add_scalar("losses/meta_policy_loss", meta_policy_loss, global_step_truth)
        writer.add_scalar("losses/option_policy_loss", option_policy_loss, global_step_truth)
        writer.add_scalar("losses/action_policy_loss", action_policy_loss, global_step_truth)
        writer.add_scalar("losses/meta_entropy", meta_entropy, global_step_truth)
        writer.add_scalar("losses/option_entropy", option_entropy, global_step_truth)
        writer.add_scalar("losses/action_entropy", action_entropy, global_step_truth)
        writer.add_scalar("losses/critic_loss_meta", critic_loss_meta, global_step_truth)
        writer.add_scalar("losses/critic_loss_option", critic_loss_option, global_step_truth)
        writer.add_scalar("losses/critic_loss_action", critic_loss_action, global_step_truth)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, args.max_grad_norm)
        self.optimizer.step()

        return total_loss

    def compute_advantages(self, rewards, values, dones, gamma, gae_lambda, next_value=0):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (torch.Tensor): Rewards from the environment.
            values (torch.Tensor): Value estimates from the critic.
            dones (torch.Tensor): Done flags indicating episode termination.
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.
            next_value (float or torch.Tensor): Value estimate for the next state after the last step.
        
        Returns:
            advantages (torch.Tensor): Computed advantages.
            returns (torch.Tensor): Computed returns.
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        next_advantage = 0

        for t in reversed(range(rewards.size(0))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            advantages[t] = next_advantage = delta + gamma * gae_lambda * next_advantage * mask
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        return advantages, returns

