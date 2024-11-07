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
    
    
class OptionCriticAgent(nn.Module):
    def __init__(self, input_channels, num_options, num_actions, hidden_dim, gamma=0.99, learning_rate=0.001):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Option over policy and intra-option policy use the same architecture
        self.policy_over_options = ProcGenActor(input_channels, num_options, num_actions, hidden_dim)
        self.intra_option_policy = ProcGenIntraActor(input_channels, num_actions, num_options, hidden_dim)
        
        # Critic network for Q-value estimation
        self.critic = ProcGenCritic(input_channels, num_options, num_actions, hidden_dim)
        
        
        
        # termination stuff
        # State representation layers (e.g., conv + fc layers for the state)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        
        # Termination network: Input is hidden_dim + num_options (for state + option)
        self.termination = nn.Linear(hidden_dim + num_options, 1)
        
        self.all_params = list(self.policy_over_options.parameters()) + list(self.intra_option_policy.parameters()) \
            + list(self.critic.parameters()) + list(self.termination.parameters())

        # Optimizers
        self.optimizer = optim.Adam(self.all_params, lr=learning_rate)
        
        
    def forward_state_rep(self, state):
        """ Forward pass through the network to get the state representation """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # State representation
        return x
        
        
    def select_option(self, state):
        """
        Select an option based on the policy over options.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)  # Handle shape (batch_size, height, width, channels)
    
        # Get the option logits from the policy over options network
        option_logits = self.policy_over_options(state / 255.0)
    
        # Convert logits into probabilities and sample an option
        option_probs = Categorical(logits=option_logits)
        option = option_probs.sample()
    
        return option
    
    
    def select_action(self, state, option):
        """
        Select an action based on the intra-option policy for the given option.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)  # Reshape to (batch_size, channels, height, width)
    
        # Forward pass through the intra-option policy network
        action_logits = self.intra_option_policy(state / 255.0)
        
        # Use `torch.gather` to select the action logits corresponding to the chosen option for each environment
        batch_size = state.size(0)
        options = option.reshape(batch_size, 1, 1)  # Reshape options to (batch_size, 1, 1) for broadcasting
        option_action_logits = torch.gather(action_logits, 1, options.expand(-1, 1, self.num_actions)).squeeze(1)
        
        # Sample an action from the selected option's action distribution
        action_probs = Categorical(logits=option_action_logits)
        action = action_probs.sample()
    
        return action
    
    
    def termination_function(self, state, option):
        """
        Compute the termination probability for the given state and option.
        - `state`: The current state.
        - `option`: The currently active option (tensor of shape (batch_size,)).
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[-1] == 3:
            state = state.permute(0,3,1,2)
        
        state_rep = self.forward_state_rep(state/255)  # Get the state representation
        
        # One-hot encode the option (num_options classes)
        option_one_hot = F.one_hot(option, num_classes=self.num_options).float()

        # Concatenate the state representation with the one-hot encoded option
        combined_input = torch.cat([state_rep, option_one_hot], dim=-1)  # Shape: (batch_size, hidden_dim + num_options)

        # Pass through the termination layer to get the termination probability
        termination_prob = torch.sigmoid(self.termination(combined_input))  # Output: (batch_size, 1)

        return termination_prob.squeeze(-1)  # Shape: (batch_size,)

    # ----------------------- Update Methods --------------------------
    
    def compute_termination_advantages(self, state, option):
        """
        Compute the termination advantage for deciding whether to terminate the current option.
        - `state`: The current state.
        - `option`: The currently active option.
        Returns:
        - `termination_advantage`: The advantage of terminating the current option.
        """

        # Compute Q-value for the current option
        Q_option = self.compute_q_value(state, option)  # Q(s, o)

        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)

        # Compute V(s) as the expected value over all options
        option_logits = self.policy_over_options(state/255)  # Get logits for options
        option_probs = F.softmax(option_logits, dim=-1)  # Convert logits to probabilities
        Q_values = self.compute_q_values_for_all_options(state)  # Get Q-values for all options
        V_state = torch.sum(option_probs * Q_values, dim=-1)  # Compute V(s)

        # Compute termination advantage: A(s, o) = Q(s, o) - V(s)
        termination_advantage = Q_option - V_state
        return termination_advantage

    def compute_q_value(self, state, option, action=None):
        """
        Compute the Q-value for a given batch of states and corresponding options and actions.
        
        - `state`: The state observations.
        - `option`: The currently selected option (batch_size, 1).
        - `action`: The action taken under the option (optiondef oc(args):
            args.batch_size = int(args.num_envs * args.num_steps)
            args.minibatch_size = int(args.batch_size // args.num_minibatches)
            args.num_iterations = args.total_timesteps // args.batch_size
            run_name = f"OC_{args.env_id}__{args.exp_name}__{args.seed}__{datetime.now()}"
        """
            
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)
    
        # Forward pass through the critic network to get Q-values for both options and actions
        q_options, q_actions = self.critic(state / 255.0)  # Critic outputs Q-values for options and actions
    
        # Ensure `option` has the correct shape
        option = option.reshape(-1, 1)  # Reshape option to match the batch size for gathering
        
        # Create a mask for valid options (where option is not -1)
        # valid_mask = (option != -1).squeeze(1)
        
        # Replace -1 with a valid index (e.g., 0) for now
        # option = torch.where(option == -1, torch.tensor(0, device=option.device), option)
        
        # Gather the Q-values for the selected option
        q_option_value = torch.gather(q_options, 1, option).squeeze(1)
        
        # q_option_value = torch.where(valid_mask.unsqueeze(-1), q_option_value, torch.zeros_like(q_option_value))

        
        if action is not None:
            # Get Q-values for the selected action under the selected option
            action = action.reshape(-1, 1)  # Reshape action for gathering
            q_action_value = torch.gather(q_actions, 1, action).squeeze(1)
    
            return q_option_value, q_action_value  # Return both option Q-value and action Q-value
    
        return q_option_value  # Return Q-value for the selected option only
    
    
    def compute_q_values_for_all_options(self, state):
        """
        Compute the Q-values for all options in the given state.
        """
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        
        if state.shape[-1] == 3:
            state = state.permute(0, 3, 1, 2)  # Convert (batch_size, height, width, channels) to (batch_size, channels, height, width)
            
        q_options, _ = self.critic(state / 255.0)  # Only get Q-values for options, discard actions
        return q_options  # Return Q-values for all options
    

    def update_combined_all(self, states, options, actions, 
                            option_advantages, action_advantages, b_values, 
                            b_returns, global_step_truth, writer, args, ent_action_coef_now, ent_option_coef_now):
        termination_advantages = self.compute_termination_advantages(states, options)
        
        if states.shape[-1] == 3:
            states = states.permute(0, 3, 1, 2)
        
        normalized_states = states / 255
        
        # Policy over options
        option_logits = self.policy_over_options(normalized_states)
        option_probs = Categorical(logits=option_logits)
        option_entropy = option_probs.entropy().mean()  # Entropy for options
        option_entropy_bonus = ent_option_coef_now * option_entropy   # Coefficient beta controls the strength of the entropy bonus
        
        # Intra-option policy (actions)
        action_logits = self.intra_option_policy(normalized_states)
        batch_size = states.shape[0]
        chosen_action_logits = torch.gather(action_logits, 1, options.reshape(batch_size, 1, 1).expand(-1, -1, action_logits.size(-1))).squeeze(1)
        action_probs = Categorical(logits=chosen_action_logits)
        action_entropy = action_probs.entropy().mean()  # Entropy for actions
        action_entropy_bonus = ent_action_coef_now * action_entropy   # Same or different beta can be used
        
        # Termination function
        termination_probs = self.termination_function(states, options)
        termination_loss = -(termination_advantages * (1 - termination_probs)).mean()
        
        # Calculate losses
        option_log_probs = option_probs.log_prob(options)
        action_log_probs = action_probs.log_prob(actions)
        option_policy_loss = -((option_log_probs * option_advantages).mean() + option_entropy_bonus)
        action_policy_loss = -((action_log_probs * action_advantages).mean() + action_entropy_bonus)
        policy_loss = option_policy_loss + action_policy_loss
                      
        critic_loss = 0.5*((b_values-b_returns)**2).mean()
        combined_loss = policy_loss + 0.5*critic_loss + termination_loss
        
        writer.add_scalar("losses/joined_policy_loss", policy_loss, global_step_truth)
        writer.add_scalar("losses/action_policy_loss", action_policy_loss, global_step_truth)
        writer.add_scalar("losses/option_policy_loss", option_policy_loss, global_step_truth)
        writer.add_scalar("losses/critic_loss", critic_loss, global_step_truth)
        writer.add_scalar("losses/termination_loss", termination_loss, global_step_truth)
        writer.add_scalar("losses/action_entropy", action_entropy, global_step_truth)
        writer.add_scalar("losses/option_entropy", option_entropy, global_step_truth)
        
        # Backward pass
        self.optimizer.zero_grad()
        combined_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, args.max_grad_norm)  # Apply gradient clipping
        self.optimizer.step()
        
        return combined_loss
