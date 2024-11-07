#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:10:43 2024

@author: x4nno
"""

# sys.path.append("/home/x4nno/Documents/PhD/FRACOs_vg")
import sys
sys.path.append('.')
sys.path.append('..')

import torch
from torch import nn
import torch.nn.functional as F

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    

import numpy as np
import torch.nn as nn

class DefaultActor(nn.Module):
    def __init__(self, observation_shape, num_actions, dropout_prob=0.4):
        super(DefaultActor, self).__init__()
        initial_shape = np.prod(observation_shape)
        
        self.layer_1 = layer_init(nn.Linear(in_features=initial_shape, out_features=64))
        self.layer_2 = layer_init(nn.Linear(in_features=64, out_features=128))
        self.layer_3 = layer_init(nn.Linear(in_features=128, out_features=256))
        self.layer_4 = layer_init(nn.Linear(in_features=256, out_features=256))
        self.layer_5 = layer_init(nn.Linear(in_features=256, out_features=256))
        self.layer_6 = layer_init(nn.Linear(in_features=256, out_features=num_actions), std=0.01)
        
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_prob)
        
    def forward(self, x):
        x = self.tanh(self.layer_1(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_2(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_3(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_4(x))
        x = self.dropout(x)
        x = self.tanh(self.layer_5(x))
        x = self.dropout(x)
        x = self.layer_6(x)
        return x
    
class ProcGenActor(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim):
        super(ProcGenActor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    
    
class ProcGenCritic(nn.Module):
    def __init__(self, input_channels, num_actions, hidden_dim):
        super(ProcGenCritic, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state): #, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten the tensor
        
        # x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value

# Helper function for layer initialization
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer



class DefaultInitClassifier(nn.Module):
    def __init__(self, input_size):
        super(DefaultInitClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    
    
class DefaultInitClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(DefaultInitClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Assuming 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class MultiClassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiClassClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x