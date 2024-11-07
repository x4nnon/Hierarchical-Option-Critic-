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

class ProcGenActor(nn.Module):
    def __init__(self, input_channels, num_options, num_actions, hidden_dim):
        super(ProcGenActor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_options)  # Output for both options and primitive actions

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits for options + primitive actions
        return x


class ProcGenIntraActor(nn.Module):
    def __init__(self, input_channels, num_actions, num_options, hidden_dim):
        self.num_options = num_options
        self.num_actions = num_actions
        super(ProcGenIntraActor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions * num_options)  # Output logits for all actions and options
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.reshape(-1, self.num_options, self.num_actions)  # Reshape to (batch_size, num_options, num_actions)
        return x    

    
class ProcGenCritic(nn.Module):
    def __init__(self, input_channels, num_options, num_actions, hidden_dim):
        super(ProcGenCritic, self).__init__()
        self.num_options = num_options
        self.num_actions = num_actions

        # Convolutional layers for state representation
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        
        # Output Q-values for both options and actions
        self.fc_options = nn.Linear(hidden_dim, num_options)  # Q-values for options
        self.fc_actions = nn.Linear(hidden_dim, num_actions)  # Q-values for actions
        

    def forward(self, state):
        # Forward pass through convolutional layers for state representation
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        x = F.relu(self.fc1(x))  # Final hidden representation of the state

        # Compute Q-values for both options and primitive actions
        q_options = self.fc_options(x)  # Shape: (batch_size, num_options)
        q_actions = self.fc_actions(x)  # Shape: (batch_size, num_actions)

        return q_options, q_actions  # Return Q-values for options and actions
    
class ProcGenMetaActor(nn.Module):
    def __init__(self, input_channels, num_meta_options, hidden_dim):
        super(ProcGenMetaActor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_meta_options)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class HOCAgent(nn.Module):
    def __init__(self, input_channels, num_meta_options, num_options, num_actions, hidden_dim, gamma=0.99, learning_rate=0.001):
        super().__init__()
        self.num_meta_options = num_meta_options
        self.num_options = num_options
        self.num_actions = num_actions
        self.gamma = gamma

        # Meta-option policy
        self.meta_option_policy = ProcGenMetaActor(input_channels, num_meta_options, hidden_dim)
        # Policy over options
        self.policy_over_options = ProcGenActor(input_channels, num_options, num_actions, hidden_dim)
        # Intra-option policy
        self.intra_option_policy = ProcGenIntraActor(input_channels, num_actions, num_options, hidden_dim)
        # Critic for option-value estimation
        self.critic = ProcGenCritic(input_channels, num_options, num_actions, hidden_dim)
        # Critic for meta-option-value estimation
        self.meta_critic = ProcGenCritic(input_channels, num_meta_options, num_actions, hidden_dim)

        # Termination networks for options and meta-options
        self.option_termination = nn.Linear(hidden_dim + num_options, 1)
        self.meta_termination = nn.Linear(hidden_dim + num_meta_options, 1)

        # Collect all parameters for joint optimization
        self.all_params = list(self.meta_option_policy.parameters()) + list(self.policy_over_options.parameters()) \
                          + list(self.intra_option_policy.parameters()) + list(self.critic.parameters()) \
                          + list(self.meta_critic.parameters()) + list(self.option_termination.parameters()) \
                          + list(self.meta_termination.parameters())
        self.optimizer = optim.Adam(self.all_params, lr=learning_rate)
        
        
    def forward_state_rep(self, state):
        """
        Forward pass to compute the state representation shared across policies.
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return x
        
        
    def select_meta_option(self, state):
        """
        Select a meta-option using the meta-option policy.
        """
        state_rep = self.forward_state_rep(state)
        meta_option_logits = self.meta_option_policy(state_rep)
        meta_option_probs = Categorical(logits=meta_option_logits)
        meta_option = meta_option_probs.sample()
        return meta_option

    def select_option(self, state, meta_option):
        """
        Select an option based on the current meta-option.
        """
        state_rep = self.forward_state_rep(state)
        option_logits = self.policy_over_options(state_rep)
        option_probs = Categorical(logits=option_logits)
        option = option_probs.sample()
        return option

    def select_action(self, state, option):
        """
        Select an action using the intra-option policy for the given option.
        """
        state_rep = self.forward_state_rep(state)
        action_logits = self.intra_option_policy(state_rep)
        batch_size = state_rep.size(0)
        option_indices = option.view(batch_size, 1, 1)
        action_logits_for_option = torch.gather(action_logits, 1, option_indices.expand(-1, -1, self.num_actions)).squeeze(1)
        action_probs = Categorical(logits=action_logits_for_option)
        action = action_probs.sample()
        return action

    def termination_function_meta(self, state, meta_option):
        """
        Compute termination probability for meta-options.
        """
        state_rep = self.forward_state_rep(state)
        meta_option_one_hot = F.one_hot(meta_option, num_classes=self.num_meta_options).float()
        combined_input = torch.cat([state_rep, meta_option_one_hot], dim=-1)
        termination_prob = torch.sigmoid(self.meta_termination(combined_input))
        return termination_prob.squeeze(-1)

    def termination_function_option(self, state, option):
        """
        Compute termination probability for regular options.
        """
        state_rep = self.forward_state_rep(state)
        option_one_hot = F.one_hot(option, num_classes=self.num_options).float()
        combined_input = torch.cat([state_rep, option_one_hot], dim=-1)
        termination_prob = torch.sigmoid(self.option_termination(combined_input))
        return termination_prob.squeeze(-1)

    def compute_q_value_meta(self, state, meta_option):
        """
        Compute the Q-value for the given state and meta-option.
        """
        state_rep = self.forward_state_rep(state)
        q_meta_options, _ = self.meta_critic(state_rep)
        q_meta_option_value = torch.gather(q_meta_options, 1, meta_option.unsqueeze(1)).squeeze(1)
        return q_meta_option_value

    def compute_q_value(self, state, option):
        """
        Compute the Q-value for the given state and regular option.
        """
        state_rep = self.forward_state_rep(state)
        q_options, _ = self.critic(state_rep)
        q_option_value = torch.gather(q_options, 1, option.unsqueeze(1)).squeeze(1)
        return q_option_value
    

    def update_function(self, batch, args, writer, global_step_truth):
        states, actions, options, meta_options, rewards, dones = batch

        # Compute the current state representation
        state_reps = self.forward_state_rep(states)

        # Compute Q-values for meta-options and options
        q_meta_values, _ = self.meta_critic(state_reps)
        q_option_values, _ = self.critic(state_reps)

        # Compute returns and advantages using GAE
        meta_advantages, meta_returns = self.compute_advantages(
            rewards, q_meta_values, dones, args.gamma, args.gae_lambda
        )
        option_advantages, option_returns = self.compute_advantages(
            rewards, q_option_values, dones, args.gamma, args.gae_lambda
        )

        # Compute policy losses for meta-options
        meta_option_logits = self.meta_option_policy(state_reps)
        meta_option_probs = Categorical(logits=meta_option_logits)
        meta_log_probs = meta_option_probs.log_prob(meta_options)
        meta_policy_loss = -(meta_log_probs * meta_advantages).mean()
        meta_entropy = meta_option_probs.entropy().mean()
        meta_policy_loss -= args.ent_coef * meta_entropy

        # Compute policy losses for options
        option_logits = self.policy_over_options(state_reps)
        option_probs = Categorical(logits=option_logits)
        option_log_probs = option_probs.log_prob(options)
        option_policy_loss = -(option_log_probs * option_advantages).mean()
        option_entropy = option_probs.entropy().mean()
        option_policy_loss -= args.ent_coef * option_entropy

        # Compute action policy losses
        action_logits = self.intra_option_policy(state_reps)
        chosen_action_logits = torch.gather(action_logits, 1, options.view(-1, 1, 1).expand(-1, -1, self.num_actions)).squeeze(1)
        action_probs = Categorical(logits=chosen_action_logits)
        action_log_probs = action_probs.log_prob(actions)
        action_policy_loss = -(action_log_probs * option_advantages).mean()
        action_entropy = action_probs.entropy().mean()
        action_policy_loss -= args.ent_coef * action_entropy

        # Compute termination losses
        termination_probs_meta = self.termination_function_meta(state_reps, meta_options)
        termination_loss_meta = -(meta_advantages * (1 - termination_probs_meta)).mean()

        termination_probs_option = self.termination_function_option(state_reps, options)
        termination_loss_option = -(option_advantages * (1 - termination_probs_option)).mean()

        # Compute critic losses
        critic_loss_meta = 0.5 * ((q_meta_values - meta_returns) ** 2).mean()
        critic_loss_option = 0.5 * ((q_option_values - option_returns) ** 2).mean()

        # Total loss
        total_loss = meta_policy_loss + option_policy_loss + action_policy_loss + \
                    termination_loss_meta + termination_loss_option + \
                    critic_loss_meta + critic_loss_option

        # Logging
        writer.add_scalar("losses/total_loss", total_loss, global_step_truth)
        writer.add_scalar("losses/meta_policy_loss", meta_policy_loss, global_step_truth)
        writer.add_scalar("losses/option_policy_loss", option_policy_loss, global_step_truth)
        writer.add_scalar("losses/action_policy_loss", action_policy_loss, global_step_truth)
        writer.add_scalar("losses/meta_entropy", meta_entropy, global_step_truth)
        writer.add_scalar("losses/option_entropy", option_entropy, global_step_truth)
        writer.add_scalar("losses/action_entropy", action_entropy, global_step_truth)

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, args.max_grad_norm)
        self.optimizer.step()

        return total_loss

