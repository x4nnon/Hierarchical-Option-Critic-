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

# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


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
        v_value = self.fc2(state_rep)
        return v_value
    

class ProcGenActionQ(nn.Module):
    def __init__(self, num_actions, hidden_dim):
        super(ProcGenActionQ, self).__init__()
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_actions), std=1)

    def forward(self, state_rep):
        q_values = self.fc2(state_rep)
        return q_values

class ProcGenOptionQ(nn.Module):
    def __init__(self, input_channels, num_options, hidden_dim):
        super(ProcGenOptionQ, self).__init__()
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_options), std=1)

    def forward(self, state_rep):
        option_q_values = self.fc2(state_rep)
        return option_q_values

class ProcGenMetaOptionQ(nn.Module):
    def __init__(self, input_channels, num_meta_options, hidden_dim):
        super(ProcGenMetaOptionQ, self).__init__()
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_meta_options), std=1)

    def forward(self, state_rep):
        meta_option_q_values = self.fc2(state_rep)
        return meta_option_q_values

class TerminationNet(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(TerminationNet, self).__init__()
        self.fc = layer_init(nn.Linear(hidden_dim, 1))

    def forward(self, state_rep):
        x = self.fc(state_rep)
        termination_prob = torch.sigmoid(x)
        return termination_prob
    

class MetaOptionActor(nn.Module):
    def __init__(self, input_channels, num_meta_options, hidden_dim):
        super(MetaOptionActor, self).__init__()
        self.fc2 = layer_init(nn.Linear(hidden_dim, num_meta_options))

    def forward(self, state_rep):
        x = self.fc2(state_rep)
        return x


class HOCAgent(nn.Module):
    def __init__(self, input_channels, num_meta_options, num_options, num_actions, hidden_dim, envs,
                  gamma=0.99, deliberation_cost=0.05, learning_rate=0.001):
        super().__init__()
        self.num_meta_options = num_meta_options
        self.num_options = num_options
        self.num_actions = num_actions
        self.gamma = gamma
        self.deliberation_cost = deliberation_cost
        
        # Meta-option policy
        self.meta_policy = MetaOptionActor(input_channels, num_meta_options, hidden_dim).to(device)
        
        # Option over policy and intra-option policies
        self.policy_over_options = [ProcGenActor(input_channels, num_options, hidden_dim).to(device) for _ in range(num_meta_options)]
        self.intra_option_policies = [ProcGenIntraActor(input_channels, num_actions, hidden_dim).to(device) for _ in range(num_options)]
        
        # Critic network for Q-value estimation
        self.critic = ProcGenCritic(input_channels, hidden_dim).to(device)
        
        # Termination networks
        self.terminations = [TerminationNet(input_channels, hidden_dim).to(device) for _ in range(num_options)]
        self.meta_terminations = [TerminationNet(input_channels, hidden_dim).to(device) for _ in range(num_meta_options)]

        # Q networks (not used in our implementation -- possible to test this vs the GAE for all, but training is stable with GAE)
        # self.action_q = ProcGenActionQ(num_actions, hidden_dim).to(device)
        # self.option_q = ProcGenOptionQ(num_options, hidden_dim).to(device)
        # self.meta_option_q = ProcGenMetaOptionQ(num_meta_options, hidden_dim).to(device)

        h, w, c = envs.single_observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.state_representation = nn.Sequential(*conv_seqs).to(device)

        # self.state_representation = ConvSequence(input_channels, 32).to(device)
        
        self.all_params = (
            list(self.meta_policy.parameters()) +
            [param for iop in self.policy_over_options for param in iop.parameters()] +
            [param for iop in self.intra_option_policies for param in iop.parameters()] +
            list(self.critic.parameters()) +
            [param for term in self.terminations for param in term.parameters()] +
            [param for term in self.meta_terminations for param in term.parameters()] +
            list(self.state_representation.parameters())
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.all_params, lr=learning_rate, weight_decay=1e-5)

    def compute_all_termination_advantage(self, state_rep, options):
        """
        This is not used in our implementation as we use GAE for all our advantages and allow a mask to handle gradients
        """
        qa_values = self.action_q(state_rep)
        qo_values = self.option_q(state_rep)
        qmo_values = self.meta_option_q(state_rep)
        
        v_values = self.critic(state_rep)
        action_advantage = qa_values.gather(1, options.unsqueeze(-1)).squeeze(-1) - v_values.squeeze(-1)
        option_advantage = qo_values.gather(1, options.unsqueeze(-1)).squeeze(-1) - v_values.squeeze(-1)
        meta_option_advantage = qmo_values.gather(1, options.unsqueeze(-1)).squeeze(-1) - v_values.squeeze(-1)
        return action_advantage, option_advantage, meta_option_advantage

    def select_meta_option(self, states, meta_options_old=None):
        meta_option_logits = self.meta_policy(states)
        meta_option_probs = Categorical(logits=meta_option_logits)
        if meta_options_old is None:
            meta_options = meta_option_probs.sample()
        else:
            meta_options = meta_options_old
        log_probs = meta_option_probs.log_prob(meta_options)
        entropy = meta_option_probs.entropy()

        return meta_options, log_probs, entropy


    def select_option(self, states, current_meta_options, options_old=None):
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
        option_logits_list = torch.zeros(states.size(0), self.num_options, device=states.device)
        for meta_option_idx in range(self.num_meta_options):
            mask = (current_meta_options == meta_option_idx)
            if mask.sum() > 0:
                selected_states = states[mask]
                option_logits = self.policy_over_options[meta_option_idx](selected_states)
                option_logits_list[mask] = option_logits
        
        # Convert logits into probabilities and create a Categorical distribution
        option_probs = Categorical(logits=option_logits_list)
    
        # If no preselected options are provided, sample new options from the distribution
        if options_old is None:
            options = option_probs.sample()
        else:
            options = options_old
    
        # Compute log probabilities and entropy of the selected options
        log_probs = option_probs.log_prob(options)
        entropy = option_probs.entropy()
    
        return options, log_probs, entropy
    
    
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

    def meta_termination_function(self, states, meta_options):
        batch_size = states.size(0)
        termination_probs = torch.zeros(batch_size, device=states.device)
        for meta_option_idx in range(self.num_meta_options):
            meta_option_mask = (meta_options == meta_option_idx)
            if meta_option_mask.sum() > 0:
                selected_states = states[meta_option_mask]
                termination_output = self.meta_terminations[meta_option_idx](selected_states)
                termination_probs[meta_option_mask] = termination_output.squeeze(-1)
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

        # Save the meta policy
        torch.save(self.meta_policy.state_dict(), os.path.join(save_path, 'meta_policy.pth'))

        # Save the policy-over-options
        for i, policy_over_option in enumerate(self.policy_over_options):
            torch.save(policy_over_option.state_dict(), os.path.join(save_path, f'policy_over_option_{i}.pth'))
        
        # Save intra-option policies
        for i, intra_option_policy in enumerate(self.intra_option_policies):
            torch.save(intra_option_policy.state_dict(), os.path.join(save_path, f'intra_option_policy_{i}.pth'))
        
        # Save the critic network
        torch.save(self.critic.state_dict(), os.path.join(save_path, 'critic.pth'))
        
        # Save termination networks
        for i, termination in enumerate(self.terminations):
            torch.save(termination.state_dict(), os.path.join(save_path, f'termination_{i}.pth'))
        for i, meta_termination in enumerate(self.meta_terminations):
            torch.save(meta_termination.state_dict(), os.path.join(save_path, f'meta_termination_{i}.pth'))


    def load(self, load_path, exclude_meta=True): 
        # Load the state representation
        
        self.state_representation.load_state_dict(torch.load(os.path.join(load_path, 'state_representation.pth')))
        
        # Load the policy-over-options
        if not exclude_meta:
            print("Loading the meta policy")
            self.meta_policy.load_state_dict(torch.load(os.path.join(load_path, 'meta_policy.pth')))

        for i, policy_over_option in enumerate(self.policy_over_options):
            policy_over_option.load_state_dict(torch.load(os.path.join(load_path, f'policy_over_option_{i}.pth')))
        
        # Load intra-option policies
        for i, intra_option_policy in enumerate(self.intra_option_policies):
            intra_option_policy.load_state_dict(torch.load(os.path.join(load_path, f'intra_option_policy_{i}.pth')))
        
        # Load the critic network
        self.critic.load_state_dict(torch.load(os.path.join(load_path, 'critic.pth')))
        
        # Load termination networks
        for i, termination in enumerate(self.terminations):
            termination.load_state_dict(torch.load(os.path.join(load_path, f'termination_{i}.pth')))
        for i, meta_termination in enumerate(self.meta_terminations):
            meta_termination.load_state_dict(torch.load(os.path.join(load_path, f'meta_termination_{i}.pth')))  
        
