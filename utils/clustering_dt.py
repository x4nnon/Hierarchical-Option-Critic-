#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:13:10 2024

@author: x4nno
"""
from dataclasses import dataclass
import tyro
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Model, GPT2Config
import pickle
import os
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset


if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

@dataclass
class Args:
    
    # Algorithm specific arguments
    env_id: str = "procgen-coinrun" 
    """the id of the environment MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun,
    atari:BreakoutNoFrameskip-v4, highway:highway-fast-v0"""
    failure_min: float = 1
    max_clusters_per_clusterer: int = 10
    incremental: bool = True ## !!!CHANGE THIS BEFORE RUNNING AGAIN TO FALSE. Doesn't refactor if true.
    current_depth: int = 0
    min_PI_score: float = 0.1
    rm_fail: bool = True ## removes the failure trajectories instead of managing with a PI score    
    chain_length: int = 3
    min_cluster_size: int = 10
    NN_epochs: int = 10
    


def calculate_metrics(output, target, num_classes):
    """
    Calculate accuracy, precision, recall, and F1-score.
    """
    # Get the predicted class by taking the argmax
    _, predicted = torch.max(output, 2)
    
    # Flatten the tensors
    predicted = predicted.view(-1)
    target = target.view(-1)
    
    accuracy = accuracy_score(target.cpu(), predicted.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(target.cpu(), predicted.cpu(), average='weighted', labels=range(num_classes))

    return accuracy, precision, recall, f1

def top_k_accuracy(output, target, k=3):
    """
    Calculate top-k accuracy.
    """
    _, top_k_predictions = output.topk(k, dim=2)
    
    target_expanded = target.unsqueeze(2).expand_as(top_k_predictions)
    correct = top_k_predictions.eq(target_expanded).sum().item()

    return correct / target.numel()



    
def remove_everything_in_folder(folder_path):
    """
    Remove all files and subdirectories in a folder.

    Args:
    - folder_path (str): Path of the folder.

    Returns:
    - None
    """
    # List all files and directories in the folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Remove all filesremove_everything_in_folder
        for file in files:
            file_path = os.path.join(root, file)
            if "a/" not in file_path:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
        # Remove all subdirectories
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if "/a" not in dir_path:
                os.rmdir(dir_path)
                print(f"Removed directory: {dir_path}")
    
    print(f"All contents in folder '{folder_path}' have been removed.")


def get_all_s_f_index(all_fractures, ep_rewards, failure_std_threshold,
                      use_std=True, failure_min=None):
    
    """Returns the a list of 1 for success and 0 for failure for every fracture """
    
    rew_mean = np.mean(ep_rewards) 
    rew_std = np.std(ep_rewards)
    
    if use_std:
        failure_threshold = rew_mean - failure_std_threshold*rew_std
    else:
        failure_threshold = failure_min
    
    failure_indexes = np.where(np.asarray(ep_rewards) < failure_threshold)[0]
    
    all_s_f = []
    for i in range(len(all_fractures)):
        for j in all_fractures[i]:
            if i in failure_indexes:
                all_s_f.append(0)
            else:
                all_s_f.append(1)
                
    return all_s_f


def create_fractures(trajectories, chain_length, a_pre_enc=False, fracos_agent=None):
    all_fractures = []
    corre_traj = []
    for trajectory in trajectories:
        move_count=0
        for move in trajectory:
            state_count = 0
            for state in move: 
                if torch.is_tensor(state):
                    trajectory[move_count][state_count] = state.cpu().detach()
                state_count += 1
            move_count += 1
            
        trajectory = np.array(trajectory, dtype=object)
        states = trajectory[:,0]
        state_list = []
        for state in states:
            
            
            state_list.append(state)
            
        state_list_arr = np.stack(state_list)
        
        obs = state_list_arr[:-(chain_length)]
        
        # here is where the vae needs to go!
        
        if "procgen" in args.env_id:
            # process the obs 
            obs = np.transpose(obs, (0, 3, 1, 2 ))
            obs.astype(np.float32) / 255.0
            obs = torch.from_numpy(obs).to(device)
            
        
        actions1 = trajectory[:-(chain_length),1]
        if a_pre_enc:
            actions1 = np.array([fracos_agent.cypher[int(item[0])] for item in actions1])
        actions1 = np.asarray(actions1)
        actions1 = np.stack(actions1)
        
        obs = obs.cpu().numpy()
        
        frac = []
        for i in range(len(obs)):
            frac.append([obs[i], actions1[i][0]])
            
        
        for b in range(1,chain_length):
            n_actions = trajectory[b:-(chain_length-b),1]
                
            if a_pre_enc:
                n_actions = np.array([fracos_agent.cypher[int(item[0])] for item in n_actions])
                
            for j in range(len(obs)):
                frac[j].append(n_actions[j][0])
        
        # frac = frac.tolist()
        
        all_fractures.append(frac)
        
    return all_fractures


class CNNEncoder(nn.Module):
    def __init__(self, image_channels, hidden_dim):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 8 * 8, hidden_dim)
        self.dropout = nn.Dropout(0.5)  # Dropout layer

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, image_channels, action_dim, hidden_dim, max_length, n_actions, num_actions):
        super(DecisionTransformer, self).__init__()
        self.cnn_encoder = CNNEncoder(image_channels, hidden_dim)
        config = GPT2Config(
            n_embd=hidden_dim,
            n_layer=6,
            n_head=8,
            n_positions=max_length,
            n_ctx=max_length,
        )
        self.transformer = GPT2Model(config)
        self.output_layer = nn.Linear(hidden_dim, num_actions)
        self.n_actions = n_actions
        self.num_actions = num_actions
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))

    def forward(self, images):
        batch_size = images.size(0)
        state_embeds = self.cnn_encoder(images)  # Shape: (batch_size, hidden_dim)
        
        # Repeat the state embedding n times and add positional encoding
        state_embeds = state_embeds.unsqueeze(1).repeat(1, self.n_actions, 1)  # Shape: (batch_size, n_actions, hidden_dim)
        state_embeds = state_embeds + self.positional_encoding[:, :self.n_actions, :]
        
        transformer_output = self.transformer(inputs_embeds=state_embeds).last_hidden_state
        return self.output_layer(transformer_output).view(batch_size, self.n_actions, self.num_actions)


class TrajectoryDataset(Dataset):
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        return state, action


if __name__ == "__main__":
    
    args = tyro.cli(Args)
    
    if not args.incremental:
        remove_everything_in_folder(f"fracos_clusters/{args.env_id}")
    
    env_name = args.env_id

    saved_traj_dir = f"trajectories/e2e_traj/{args.env_id}_D{args.current_depth}/"
        
    traj_path = saved_traj_dir+"trajs.p"
    rew_path = saved_traj_dir+"rews.p"
    traj_content = pickle.load(open(traj_path, "rb"))
    rew_content = pickle.load(open(rew_path, "rb"))
    
    all_trajectories = traj_content
    all_ep_rewards = rew_content
    
    ######### remove ##############
    all_trajectories = all_trajectories[:100]
    all_ep_rewards = all_ep_rewards[:100]
    ##################################
    
    if args.rm_fail:
        success_idxs = [index for index, value in enumerate(all_ep_rewards) if value > args.failure_min]
        all_trajectories = [all_trajectories[idx] for idx in success_idxs]
        all_ep_rewards = [all_ep_rewards[idx] for idx in success_idxs]
        
    # Rather than what we have done here above, why don't we do the same clustering method and then approximate those states?
        
    print("creating fractures")
    fractures = create_fractures(all_trajectories, args.chain_length)
    fractures = [item for subfrac in fractures for item in subfrac]
    # get all our obs
    
    # not needed if we use rm_fail
    all_s_f = get_all_s_f_index(all_trajectories, all_ep_rewards, failure_std_threshold=None,
                                use_std=False, failure_min=args.failure_min)
    # Assuming you have prepared your data as sequences of states and actions
    states = []
    actions = []
    for frac in fractures:
        states.append(frac[0])
        actions.append(frac[1:])
    
    print("converting to tensors")
    
    states = torch.tensor(states, dtype=torch.float32)/255 #added /255 to normalize
    actions = torch.tensor(actions, dtype=torch.long)
    
    X_train, X_test, y_train, y_test = train_test_split(states, actions, test_size=0.2, random_state=42)
    
    X_train= torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TrajectoryDataset(X_train, y_train)
    test_dataset = TrajectoryDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    print("finished converting to tensors")
    
    image_channels = 3
    n_actions = args.chain_length  # Predict the next 2 actions
    num_actions = 15
    hidden_dim = 128  # Dimension of the hidden layers
    max_length = args.chain_length  # The maximum length of the sequence in the dataset (10) for some reason on orignal
    
    model = DecisionTransformer(image_channels, 1, hidden_dim, max_length, n_actions, num_actions)

    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Include all model parameters
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_train_outputs = []
        all_train_targets = []
        for batch_states, batch_actions in train_loader:
            optimizer.zero_grad()
            output = model(batch_states)
            all_train_outputs.append(output)
            all_train_targets.append(batch_actions)
            loss = criterion(output.view(-1, num_actions), batch_actions.view(-1, 1).flatten())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        all_train_outputs = torch.cat(all_train_outputs)
        all_train_targets = torch.cat(all_train_targets)
        accuracy, precision, recall, f1 = calculate_metrics(all_train_outputs, all_train_targets, num_actions)
        top_3_acc = top_k_accuracy(all_train_outputs, all_train_targets, k=3)

        print("-------- TRAIN ---------- ")
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Top-3 Accuracy: {top_3_acc:.4f}')
    
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_val_outputs = []
            all_val_targets = []
            for batch_states, batch_actions in test_loader:
                output = model(batch_states)
                loss = criterion(output.view(-1, num_actions), batch_actions.view(-1, 1).flatten())
                val_loss += loss.item()
                all_val_outputs.append(output)
                all_val_targets.append(batch_actions)
    
            val_loss /= len(test_loader)
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_targets = torch.cat(all_val_targets)
            accuracy, precision, recall, f1 = calculate_metrics(all_val_outputs, all_val_targets, num_actions)
            top_3_acc = top_k_accuracy(all_val_outputs, all_val_targets, k=3)
    
        print("-------- VAL ---------- ")
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        print(f'Top-3 Accuracy: {top_3_acc:.4f}')
        
    os.makedirs(f"fracos_clusters/{args.env_id}/d/{args.current_depth}", exist_ok=True)
    model_save_path = f"fracos_clusters/{args.env_id}/d/{args.current_depth}/transformerModel.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    
    #### how to load later
    # Initialize the model (same architecture as before)
    # model = DecisionTransformer(image_channels, 1, hidden_dim, max_length, n_actions, num_actions)
    
    # # Load the model state dictionary
    # model.load_state_dict(torch.load(model_save_path))
    
    # # Set the model to evaluation mode
    # model.eval()