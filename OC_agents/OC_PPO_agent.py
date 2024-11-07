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
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class StateRepresentation(nn.Module):
#     def __init__(self, input_channels, hidden_dim):
#         super(StateRepresentation, self).__init__()
#         self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
#         self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)

#     def forward(self, state):
#         x = F.relu(self.conv1(state))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = x.reshape(x.size(0), -1)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         return x


# class ProcGenActor(nn.Module):
#     def __init__(self, input_channels, num_options, hidden_dim):
#         super(ProcGenActor, self).__init__()
#         # Each actor has its own state representation
#         self.state_representation = StateRepresentation(input_channels, hidden_dim)
#         # Output layer for options
#         self.fc2 = nn.Linear(hidden_dim, num_options)

#     def forward(self, state):
#         # Pass the input through its own state representation network
#         state_representation = self.state_representation(state)
#         # Compute logits for options
#         x = self.fc2(state_representation)
#         return x


# class ProcGenIntraActor(nn.Module):
#     def __init__(self, input_channels, num_actions, hidden_dim):
#         super(ProcGenIntraActor, self).__init__()
#         # Each intra-actor has its own state representation
#         self.state_representation = StateRepresentation(input_channels, hidden_dim)
#         # Output layer for actions
#         self.fc2 = nn.Linear(hidden_dim, num_actions)
#         self.num_actions = num_actions

#     def forward(self, state):
#         # Pass the input through its own state representation network
#         state_representation = self.state_representation(state)
#         # Compute logits for actions
#         x = self.fc2(state_representation)
#         x = x.view(-1, self.num_actions)
#         return x


# class ProcGenCritic(nn.Module):
#     def __init__(self, input_channels, hidden_dim):
#         super(ProcGenCritic, self).__init__()
#         # Each critic has its own state representation
#         self.state_representation = StateRepresentation(input_channels, hidden_dim)
#         # Output layer for value estimation
#         self.fc2 = nn.Linear(hidden_dim, 1)

#     def forward(self, state):
#         # Pass the input through its own state representation network
#         state_representation = self.state_representation(state)
#         # Compute value estimate
#         q_value = self.fc2(state_representation)
#         return q_value


# class TerminationNet(nn.Module):
#     def __init__(self, input_channels, hidden_dim):
#         super(TerminationNet, self).__init__()
#         # Each termination net has its own state representation
#         self.state_representation = StateRepresentation(input_channels, hidden_dim)
#         # Output layer for termination probability
#         self.termination = nn.Linear(hidden_dim, 1)

#     def forward(self, state):
#         # Pass the input through its own state representation network
#         state_representation = self.state_representation(state)
#         # Compute termination probability
#         termination_prob = torch.sigmoid(self.termination(state_representation))
#         return termination_prob
    

# class OptionCriticAgent(nn.Module):
#     def __init__(self, input_channels, num_options, num_actions, hidden_dim, gamma=0.99, learning_rate=0.001):
#         super().__init__()
#         self.num_options = num_options
#         self.num_actions = num_actions
#         self.gamma = gamma
        
#         # Option over policy and intra-option policy use the same architecture
#         self.policy_over_options = ProcGenActor(input_channels, num_options, hidden_dim).to(device)
#         self.intra_option_policies = [ProcGenIntraActor(input_channels, num_actions, hidden_dim).to(device) for i in range(num_options)]
        
#         # Critic network for Q-value estimation
#         self.critic = ProcGenCritic(input_channels, hidden_dim).to(device)
        
#         self.terminations = [TerminationNet(input_channels, hidden_dim).to(device) for i in range(num_options)]
        
#         self.all_params = (
#             list(self.policy_over_options.parameters()) +
#             [param for iop in self.intra_option_policies for param in iop.parameters()] +
#             list(self.critic.parameters()) +
#             [param for term in self.terminations for param in term.parameters()] 
#         )
#         # Optimizers
#         self.optimizer = optim.Adam(self.all_params, lr=learning_rate)
        
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StateRepresentation(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(StateRepresentation, self).__init__()
        # Initialize layers using layer_init
        self.conv1 = (nn.Conv2d(input_channels, 32, kernel_size=3, stride=2))
        self.conv2 = (nn.Conv2d(32, 64, kernel_size=3, stride=2))
        self.conv3 = (nn.Conv2d(64, 64, kernel_size=3, stride=2))
        self.fc1 = (nn.Linear(64 * 7 * 7, hidden_dim))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return x


class ProcGenActor(nn.Module):
    def __init__(self, input_channels, num_options, hidden_dim):
        super(ProcGenActor, self).__init__()
        # Initialize the output layer
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_options))

    def forward(self, state_rep):
        # Compute logits for options
        x = self.fc2(state_rep)
        return x


class ProcGenIntraActor(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim):
        super(ProcGenIntraActor, self).__init__()
        # Initialize the output layer
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_actions))
        self.num_actions = num_actions

    def forward(self, state_rep):
        # Compute logits for actions
        x = self.fc2(state_rep)
        x = x.view(-1, self.num_actions)
        return x


class ProcGenCritic(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ProcGenCritic, self).__init__()
        # Initialize the output layer
        self.fc2 = layer_init(nn.Linear(hidden_dim, 1), std=1)

    def forward(self, state_rep):
        # Compute value estimate
        q_value = self.fc2(state_rep)
        return q_value


class TerminationNet(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(TerminationNet, self).__init__()
        # Initialize the output layer
        self.termination = layer_init(nn.Linear(hidden_dim, 1))

    def forward(self, state_rep):
        # Compute termination probability
        termination_prob = torch.sigmoid(self.termination(state_rep))
        return termination_prob
    

class OptionCriticAgent(nn.Module):
    def __init__(self, input_channels, num_options, num_actions, hidden_dim, gamma=0.99, learning_rate=0.001):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        self.gamma = gamma
        
        # Option over policy and intra-option policies use the same architecture
        self.policy_over_options = ProcGenActor(input_channels, num_options, hidden_dim).to(device)
        self.intra_option_policies = [ProcGenIntraActor(input_channels, num_actions, hidden_dim).to(device) for _ in range(num_options)]
        
        # Critic network for Q-value estimation
        self.critic = ProcGenCritic(input_channels, hidden_dim).to(device)
        
        # Termination networks
        self.terminations = [TerminationNet(input_channels, hidden_dim).to(device) for _ in range(num_options)]
        
        self.state_representation = StateRepresentation(input_channels, hidden_dim)
        self.state_representation_o = StateRepresentation(input_channels, hidden_dim)
        
        self.all_params = (
            list(self.policy_over_options.parameters()) +
            [param for iop in self.intra_option_policies for param in iop.parameters()] +
            list(self.critic.parameters()) +
            [param for term in self.terminations for param in term.parameters()] +
            list(self.state_representation.parameters()) #+
            # list(self.state_representation_o.parameters())
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.all_params, lr=learning_rate)

    def select_option(self, states, options_old=None):
        """
        Select an option for each state in the batch based on the policy over options.
        
        Args:
            states (torch.Tensor): The batch of state representations.
            options_old (torch.Tensor, optional): Preselected options, if available.
        
        Returns:
            torch.Tensor: The batch of selected options.
            torch.Tensor: The log probabilities of the selected options.
            torch.Tensor: The entropy of the option distributions.
        """
        # Get the option logits for each state in the batch from the policy over options network
        option_logits = self.policy_over_options(states)
        
        # Convert logits into probabilities and create a Categorical distribution
        option_probs = Categorical(logits=option_logits)
    
        # If no preselected options are provided, sample new options from the distribution
        if options_old is None:
            options = option_probs.sample()
        else:
            options = options_old
    
        # Compute log probabilities and entropy of the selected options
        log_probs = option_probs.log_prob(options)
        entropy = option_probs.entropy()
    
        return options, log_probs, entropy
    
    
    # def select_action(self, state_reps, options, actions_old=None):
    #     """
    #     Select an action based on the intra-option policy for the given batch of states and options.
        
    #     Args:
    #         state_rep (torch.Tensor): The batch of state_rep.
    #         options (torch.Tensor): The batch of options (indices into the intra-option policies).
        
    #     Returns:
    #         torch.Tensor: The batch of selected actions.
    #     """
    #     # Ensure input states are tensors and have the correct shape

    
        
    #     actions = []
    #     logits = []
    #     entropy = []
        
    #     counter = 0
    #     # Iterate over the batch of states and options
    #     for state_rep, option in zip(state_reps, options):
    #         # Forward pass through the selected intra-option policy network
    #         action_logits = self.intra_option_policies[option](state_rep.unsqueeze(0))  # Add batch dimension
            
    #         action_probs = Categorical(logits=action_logits)
    #         entropy.append(action_probs.entropy())
    #         if actions_old is None:
    #             action = action_probs.sample()
    #         else:
    #             action = actions_old[counter]
    #         logits.append(action_probs.log_prob(action))
            
    #         actions.append(action)
            
    #         counter += 1
    
    #     # Stack actions into a single tensor to return
    #     return torch.stack(actions).flatten(), torch.stack(logits).flatten(), torch.stack(entropy)
    
    def select_action(self, states, options, actions_old=None):
        """
        Select an action based on the intra-option policy for the given batch of states and options.
        
        Args:
            states (torch.Tensor): The batch of state representations.
            options (torch.Tensor): The batch of options (indices into the intra-option policies).
            actions_old (torch.Tensor, optional): Optional actions for reuse.
        
        Returns:
            torch.Tensor: The batch of selected actions, log probabilities, and entropy.
        """
        batch_size = states.size(0)
        
        # Tensor to hold the action logits for the entire batch
        action_logits_list = torch.zeros(batch_size, self.num_actions, device=states.device)
        
        # Process all options in parallel by gathering the relevant states for each option
        for option_idx in range(self.num_options):
            mask = (options == option_idx)  # Mask to select states that have the current option
            
            if mask.sum() > 0:  # Only process if there are states with this option
                selected_states = states[mask]
                action_logits_list[mask] = self.intra_option_policies[option_idx](selected_states)
        
        # Create a Categorical distribution over the action logits
        action_probs = Categorical(logits=action_logits_list)
    
        # Sample actions from the action probabilities (or reuse the old actions if provided)
        if actions_old is None:
            actions = action_probs.sample()
        else:
            actions = actions_old
    
        # Compute the log probabilities and entropy of the selected actions
        log_probs = action_probs.log_prob(actions)
        entropy = action_probs.entropy()
    
        return actions, log_probs, entropy
    
    
    def termination_function(self, states, options):
        """
        Batch process the termination probabilities for a batch of states and options.
        
        Args:
            states (torch.Tensor): The batch of states.
            options (torch.Tensor): The batch of options (indices into the termination networks).
            
        Returns:
            torch.Tensor: The termination probabilities for each state-option pair in the batch.
        """
        batch_size = states.size(0)
        termination_probs = torch.zeros(batch_size, device=states.device)  # Initialize the batch of termination probabilities
    
        # For each option, apply the corresponding termination network to the states that have this option
        for option_idx in range(self.num_options):
            # Create a mask to identify states that are using the current option
            option_mask = (options == option_idx)
            
            if option_mask.sum() > 0:  # Process if there are any states with this option
                selected_states = states[option_mask]  # Extract the states with the current option
                termination_output = self.terminations[option_idx](selected_states)  # Apply termination network
                termination_probs[option_mask] = termination_output.squeeze(-1)  # Store the results in the correct positions
                
        return termination_probs

    
    def compute_value(self, state):
        """
        Compute the value for the given state using the critic network.
        
        Args:
            state (torch.Tensor): The input state (assumed to be a tensor).
            
        Returns:
            torch.Tensor: The estimated Q-value for the given state.
        """
        
        
        # Use the critic network to compute the value for the state
        value = self.critic(state)
        
        return value

    def save(self, save_path):
        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        # Save the state representation
        torch.save(self.state_representation.state_dict(), os.path.join(save_path, 'state_representation.pth'))
        torch.save(self.state_representation_o.state_dict(), os.path.join(save_path, 'state_representation_o.pth'))
        
        # Save the policy-over-options
        torch.save(self.policy_over_options.state_dict(), os.path.join(save_path, 'policy_over_options.pth'))
        
        # Save intra-option policies
        for i, intra_option_policy in enumerate(self.intra_option_policies):
            torch.save(intra_option_policy.state_dict(), os.path.join(save_path, f'intra_option_policy_{i}.pth'))
        
        # Save the critic network
        torch.save(self.critic.state_dict(), os.path.join(save_path, 'critic.pth'))
        
        # Save termination networks
        for i, termination in enumerate(self.terminations):
            torch.save(termination.state_dict(), os.path.join(save_path, f'termination_{i}.pth'))


    def load(self, load_path, exclude_meta=True): 
        # Load the state representation
        
        self.state_representation.load_state_dict(torch.load(os.path.join(load_path, 'state_representation.pth')))
        
        # Load the policy-over-options
        if not exclude_meta:
            print("Loading the meta policy")
            self.state_representation_o.load_state_dict(torch.load(os.path.join(load_path, 'state_representation_o.pth')))
            self.policy_over_options.load_state_dict(torch.load(os.path.join(load_path, 'policy_over_options.pth')))
        
        # Load intra-option policies
        for i, intra_option_policy in enumerate(self.intra_option_policies):
            intra_option_policy.load_state_dict(torch.load(os.path.join(load_path, f'intra_option_policy_{i}.pth')))
        
        # Load the critic network
        self.critic.load_state_dict(torch.load(os.path.join(load_path, 'critic.pth')))
        
        # Load termination networks
        for i, termination in enumerate(self.terminations):
            termination.load_state_dict(torch.load(os.path.join(load_path, f'termination_{i}.pth')))
        