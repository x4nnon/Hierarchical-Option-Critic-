#### THIS IS ONLY SET UP TO WORK WITH PROCGEN FOR NOW


import os
# sys.path.append("/home/x4nno/Documents/PhD/FRACOs_vg")
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import pickle
import matplotlib.pyplot as plt

import torch
import pickle

from dataclasses import dataclass
import tyro

import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from itertools import chain

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from utils.compression import cluster_PI_compression

from imblearn.over_sampling import SMOTE

from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset, ConcatDataset
import torch.optim as optim
from sklearn.metrics import accuracy_score

from utils.default_networks import DefaultInitClassifier, MultiClassClassifier, DefaultInitClassifierCNN

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from utils.VAE_creation_fracos import VAE_procgen
import hdbscan

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    #torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

@dataclass
class Args:
    
    # Algorithm specific arguments
    env_id: str = "procgen-starpilot" 
    """the id of the environment MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun,
    atari:BreakoutNoFrameskip-v4, highway:highway-fast-v0"""
    failure_min: float = 1
    max_clusters_per_clusterer: int = 10
    incremental: bool = False ## !!!CHANGE THIS BEFORE RUNNING AGAIN TO FALSE. Doesn't refactor if true.
    current_depth: int = 0
    min_PI_score: float = 0.1
    rm_fail: bool = True ## removes the failure trajectories instead of managing with a PI score    
    chain_length: int = 2
    min_cluster_size: int = 10
    NN_epochs: int = 10

# class CustomDataset(Dataset):
#     def __init__(self, data, labels, transform=None):
#         self.data = torch.tensor(data, dtype=torch.float32)
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         if self.transform:
#             sample = self.transform(sample)
#         return sample, self.labels[idx]

def visualize_images_per_label_separately(images, labels, num_images=16, title=None):
    """
    Visualizes a specified number of images per label in the dataset, displaying each label's images separately.

    Args:
        images (numpy.ndarray): The image data to visualize.
        labels (numpy.ndarray): The labels corresponding to the images.
        num_images (int): The number of images to display per label.
        title (str): Optional title for the plot.
    """
    # Get unique labels
    unique_labels = np.unique(labels)
    
    # Calculate the grid size for each label
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Plot images for each label
    for label in unique_labels:
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
        
        if title:
            fig.suptitle(f"{title} - Label: {label}", fontsize=16)
        
        label_indices = np.where(labels == label)[0][:num_images]
        
        for i in range(grid_size * grid_size):
            ax = axes.flatten()[i]
            if i < len(label_indices):
                img = images[label_indices[i]]
                ax.imshow(img.transpose(1, 2, 0)/255)  # Transpose the image if needed (C, H, W) -> (H, W, C)
                ax.set_title(f'Label: {label}')
            ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


class GaussianNoiseTransform:
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

def save_all_clusterings(clusterer, clusters, concat_fractures, concat_trajs, NN, model_args, method, cluster_level, env_name):
    # check if dir exist if not make it.
    if not os.path.exists("fracos_clusters/" + env_name + "/clusterers"):
        os.makedirs("fracos_clusters/" + env_name + "/clusterers")
    if not os.path.exists("fracos_clusters/" + env_name + "/clusters"):
        os.makedirs("fracos_clusters/" + env_name + "/clusters")
    if not os.path.exists("fracos_clusters/" + env_name + "/other"):
        os.makedirs("fracos_clusters/" + env_name + "/other")
    if not os.path.exists("fracos_clusters/" + env_name + "/NNs"):
        os.makedirs("fracos_clusters/" + env_name + "/NNs")
    if not os.path.exists("fracos_clusters/" + env_name + "/NN_args"):
        os.makedirs("fracos_clusters/" + env_name + "/NN_args")
    if not os.path.exists("fracos_clusters/" + env_name + "/cluster_cyphers"):
        os.makedirs("fracos_clusters/" + env_name + "/cluster_cyphers")
    if not os.path.exists("fracos_clusters/" + env_name + "/cluster_reverse_cyphers"):
        os.makedirs("fracos_clusters/" + env_name + "/cluster_reverse_cyphers")
    
    
    pickle.dump(clusterer, open("fracos_clusters/" + env_name + "/clusterers/" + "clusterer{}.p".format(cluster_level), "wb"))
    pickle.dump(clusters, open("fracos_clusters/" + env_name + "/clusters/" + "clusters{}.p".format(cluster_level), "wb"))
    pickle.dump(concat_fractures, open("fracos_clusters/" + env_name + "/other/" + "concat_fractures{}.p".format(cluster_level), "wb"))    
    pickle.dump(concat_trajs, open("fracos_clusters/" + env_name + "/other/" + "concat_trajs{}.p".format(cluster_level), "wb")) 
    if NN is not None:
        torch.save(NN.state_dict(), "fracos_clusters/"+ env_name + "/NNs/" + "NN_state_dict_{}.pth".format(cluster_level))
    pickle.dump(model_args, open("fracos_clusters/" + env_name + "/NN_args/" + "NN_args_{}.p".format(cluster_level), "wb"))

    if method is not None:
        pickle.dump(method.cypher, open("fracos_clusters/" + env_name + "/cluster_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))
        pickle.dump(method.reverse_cypher, open("fracos_clusters/" + env_name + "/cluster_reverse_cyphers/" + "cypher_{}.p".format(cluster_level), "wb"))


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
            
        # do we actually want these as normal actions not cyphered?
        actions1 = np.asarray(actions1)
        actions1 = np.stack(actions1)
        
        obs = obs.cpu().numpy()
        
        frac = []
        for i in range(len(obs)):
            frac.append([obs[i], actions1[i]])
            
        
        for b in range(1,chain_length):
            n_actions = trajectory[b:-(chain_length-b),1]
                
            if a_pre_enc:
                n_actions = np.array([fracos_agent.cypher[int(item[0])] for item in n_actions])
                
            for j in range(len(obs)):
                frac[j].append(n_actions[j])
        
        # frac = frac.tolist()
        
        all_fractures.append(frac)
        
    return all_fractures

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


def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def scale_data(data, scale_range=(0.9, 1.1)):
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], data.shape)
    return data * scale_factors

def jitter_data(data, jitter_amount=0.01):
    jitter = np.random.uniform(-jitter_amount, jitter_amount, data.shape)
    return data + jitter

def SMOTE_data(features, labels):
    smote = SMOTE()
    augmented_features, augmented_labels = smote.fit_resample(features, labels)
    return augmented_features, augmented_labels


def augment_data_pipeline(features, labels):
    if "MetaGrid" not in args.env_id:
        noise_features = add_noise(features)
        noise_labels = labels
    
        all_features = np.concatenate((features, noise_features))
        all_labels = np.concatenate((labels, noise_labels))
    
    else:
        all_features = features
        all_labels = labels
    
    all_features, all_labels = SMOTE_data(all_features, all_labels)
    
    return all_features, all_labels


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
    
def feature_extraction_pipeline(images, model, runtime=False):
    # preprocess = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    
    # # Function to preprocess image
    # def preprocess_image(img_array):
    #     img_tensor = preprocess(img_array)
    #     img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    #     return img_tensor
    
    # Extract features
    features = []
    feature_shape_found = False
    with torch.no_grad():
        i = 0
        for img_array in images:
            if i % 100 == 0:
                print (f"{i}/{len(images)}")
            i += 1
            # img_tensor = preprocess_image(img_array[:, :, :3]).to(device)  # Use first 3 channels for ResNet
            if isinstance(img_array, np.ndarray):
                if img_array.shape[0] != 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                    img_array = np.expand_dims(img_array, axis=0)
            
            img_tensor = torch.as_tensor(img_array).to(device)
            feature = model(img_tensor)
            if not feature_shape_found:
                feature_shape = feature.shape
                feature_shape_found = True
            
            if runtime:
                feature = feature.squeeze()
                return feature
            
            if feature.shape == feature_shape:
                features.append(feature.squeeze().cpu())
    
    features = np.stack([tensor.numpy() for tensor in features])
    return features
    

def vae_extraction_pipeline(images, vae, runtime=False):
    features = []
    feature_shape_found = False
    with torch.no_grad():
        i = 0
        for img_array in images:
            if i % 100 == 0:
                print (f"{i}/{len(images)}")
            i += 1
            # img_tensor = preprocess_image(img_array[:, :, :3]).to(device)  # Use first 3 channels for ResNet
            if isinstance(img_array, np.ndarray):
                if img_array.shape[0] != 3:
                    img_array = np.transpose(img_array, (2, 0, 1))
                    img_array = np.expand_dims(img_array, axis=0)
            
            img_tensor = torch.as_tensor(img_array).to(device)
            _, feature_mu, feature_logvar = vae(img_tensor/255)
            feature = vae.reparameterize(feature_mu, feature_logvar)
            if not feature_shape_found:
                feature_shape = feature.shape
                feature_shape_found = True
            
            if runtime:
                feature = feature.squeeze()
                return feature
            
            if feature.shape == feature_shape:
                features.append(feature.squeeze().cpu())
    
    features = np.stack([tensor.numpy() for tensor in features])
    return features


def get_all_states(trajectories, chain_length):
    all_obs = []
    corre_traj = []
    for trajectory in trajectories:
                    
        trajectory = np.array(trajectory, dtype=object)
        states = trajectory[:,0]
        states = np.stack(states)
        all_obs.append(states)
    all_obs = np.vstack(all_obs)
    
    return all_obs


# def NN_approx_init_states(all_obs, all_labels, num_epochs=100, confidence_threshold=0.51):
    
#     # visualize_images_per_label_separately(all_obs, all_labels)
#     # all_obs, all_labels = SMOTE_data(all_obs, all_labels) # to keep things balanced
    
#     X_train, X_test, y_train, y_test = train_test_split(all_obs, all_labels, test_size=0.2, random_state=42)
    
#     X_train = np.array(X_train)
    
#     # We need to augment our minority classes:
#     class_counts = Counter(y_train)
#     print("Class counts before balancing:", class_counts)
#     class_sizes = sorted(class_counts.values(), reverse=True)

#     max_count = max(class_counts.values())
#     print("Maximum class count:", max_count)

#     # Determine the size of the second largest class
#     second_largest_size = class_sizes[1]
#     print("Size of the second largest class:", second_largest_size)
    
#     # Prepare to store the balanced data and labels
#     balanced_data = []
#     balanced_labels = []
    
#     # Iterate through each class
#     for label in np.unique(y_train):
#         class_data = X_train[y_train == label]
#         if class_counts[label] > second_largest_size:
#             # Undersample to the size of the second largest class
#             indices = np.random.choice(len(class_data), second_largest_size, replace=False)
#             class_data = class_data[indices]
#         balanced_data.append(class_data)
#         balanced_labels.extend([label] * len(class_data))
    
#     # Concatenate the balanced data and labels
#     balanced_data = np.concatenate(balanced_data)
#     balanced_labels = np.array(balanced_labels)
    
#     # Verify the new class counts
#     balanced_class_counts = Counter(balanced_labels)
#     print("Class counts after undersampling:", balanced_class_counts)
    
    
    
#     # Augment the minority classes
#     augmented_data = []
#     augmented_labels = []
#     noise_transform = GaussianNoiseTransform(mean=0, std=0.5)
#     for label in np.unique(y_train):
#         class_data = balanced_data[balanced_labels == label]
#         num_samples_needed = second_largest_size - len(class_data)
        
#         for _ in range(num_samples_needed):
#             idx = np.random.choice(len(class_data))
#             sample = class_data[idx] / 255.0  # Normalize to [0, 1]
#             augmented_sample = noise_transform(torch.tensor(sample, dtype=torch.float32))
#             augmented_sample = torch.clamp(augmented_sample, 0, 1)  # Clip to [0, 1]
#             augmented_data.append(augmented_sample.numpy() * 255.0)  # Denormalize back to [0, 255]
#             augmented_labels.append(label)
    
#     # Convert augmented data and labels to numpy arrays
#     augmented_data = np.array(augmented_data)
#     augmented_labels = np.array(augmented_labels)
    
#     # visualize_images_per_label_separately(augmented_data, augmented_labels)
    
#     # Combine original and augmented data
#     final_balanced_data = np.concatenate((balanced_data, augmented_data))
#     final_balanced_labels = np.concatenate((balanced_labels, augmented_labels))
    
#     # Verify the final class counts
#     final_class_counts = Counter(final_balanced_labels)
#     print("Class counts after balancing:", final_class_counts)
    
    
#     balanced_train_dataset = CustomDataset(final_balanced_data, final_balanced_labels)
#     balanced_train_loader = DataLoader(balanced_train_dataset, batch_size=128, shuffle=True)
#     test_dataset = CustomDataset(X_test, y_test)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     num_classes = max(y_train)+1
#     model = DefaultInitClassifierCNN(num_classes)
    
#     criterion = nn.CrossEntropyLoss().to("cuda")
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     model.to("cuda")
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(balanced_train_loader):
#             optimizer.zero_grad()
#             inputs = inputs.to("cuda")
#             labels = labels.to("cuda")
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
            
#             if i % 100 == 9:    # Print every 100 batches
#                 print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss / 10:.3f}")
#                 running_loss = 0.0
                
#                 model.eval()
#                 correct = 0
#                 total = 0
#                 confident_total = 0
            
#                 with torch.no_grad():
#                     for inputs, labels in test_loader:
#                         inputs, labels = inputs.to(device), labels.to(device)
#                         outputs = model(inputs)
            
#                         # Calculate softmax probabilities
#                         probabilities = F.softmax(outputs, dim=1)
#                         max_probs, predicted = torch.max(probabilities, dim=1)
                        
            
#                         # Only consider predictions with high confidence
#                         confident_indices = (max_probs > confidence_threshold) & (predicted != args.max_clusters_per_clusterer)
#                         confident_total += confident_indices.sum().item()
#                         correct += (predicted[confident_indices] == labels[confident_indices]).sum().item()
#                         total += labels.size(0)
            
#                 if confident_total == 0:
#                     print("No predictions met the confidence threshold.")
#                 else:
#                     accuracy = 100 * correct / confident_total
#                     confidence_coverage = 100 * confident_total / total
#                     print(f"Accuracy of the model on the test images with confidence threshold {confidence_threshold}: {accuracy:.2f}%")
#                     print(f"Coverage of confident predictions: {confidence_coverage:.2f}%")

                
#                 correct = 0
#                 total = 0
#                 with torch.no_grad():
#                     countt = 0
#                     for inputs, labels in balanced_train_loader:
#                         while countt < 100:
                            
#                             inputs = inputs.to("cuda")
#                             labels = labels.to("cuda")
#                             outputs = model(inputs)
#                             _, predicted = torch.max(outputs, 1)
#                             total += labels.size(0)
#                             correct += (predicted == labels).sum().item()
#                             countt += 1
                
#                 print(f"Accuracy of the model on the train images: {100 * correct / total:.2f}%")
#                 model.train()
    
#     print("Finished Training")
    
#     model.eval()
#     correct = 0
#     total = 0
#     confident_total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)

#             # Calculate softmax probabilities
#             probabilities = F.softmax(outputs, dim=1)
#             max_probs, predicted = torch.max(probabilities, dim=1)

#             # Only consider predictions with high confidence
#             confident_indices = (max_probs > confidence_threshold) & (predicted != args.max_clusters_per_clusterer)
#             confident_total += confident_indices.sum().item()
#             correct += (predicted[confident_indices] == labels[confident_indices]).sum().item()
#             total += labels.size(0)

#     if confident_total == 0:
#         print("No predictions met the confidence threshold.")
#     else:
#         accuracy = 100 * correct / confident_total
#         confidence_coverage = 100 * confident_total / total
#         print(f"Accuracy of the model on the test images with confidence threshold {confidence_threshold}: {accuracy:.2f}%")
#         print(f"Coverage of confident predictions: {confidence_coverage:.2f}%")

        
    
#     return model
    
class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample.permute(1, 2, 0).numpy())  # Convert to (H, W, C) for transform
        # sample = torch.tensor(sample).permute(2, 0, 1)  # Convert back to (C, H, W)
        return sample, self.labels[idx]

def evaluate_model_with_confidence_and_exclusions(model, test_loader, device, confidence_threshold=0.9, exclude_label=10):
    model.eval()
    correct = 0
    total = 0
    confident_total = 0
    excluded_total = 0  # To track how many predictions were excluded

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # Calculate softmax probabilities
            probabilities = F.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probabilities, dim=1)

            # Only consider predictions with high confidence and exclude predictions equal to exclude_label
            confident_indices = (max_probs > confidence_threshold) & (predicted != exclude_label)
            confident_total += confident_indices.sum().item()
            correct += (predicted[confident_indices] == labels[confident_indices]).sum().item()

            # Track excluded predictions
            excluded_indices = (predicted == exclude_label)
            excluded_total += excluded_indices.sum().item()

            total += labels.size(0)

    if confident_total == 0:
        print("No predictions met the confidence threshold.")
        return 0

    accuracy = 100 * correct / confident_total
    confidence_coverage = 100 * confident_total / total
    excluded_percentage = 100 * excluded_total / total

    print(f"Accuracy of the model on the test images with confidence threshold {confidence_threshold}: {accuracy:.2f}%")
    print(f"Coverage of confident predictions: {confidence_coverage:.2f}%")
    print(f"Percentage of predictions excluded due to label {exclude_label}: {excluded_percentage:.2f}%")

#### ViT stuff

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, H*W/patch_size^2)
        x = x.transpose(1, 2)  # (B, H*W/patch_size^2, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, num_patches):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.pos_embed

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(embed_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=1)
        x = (x * attn_weights).sum(dim=1)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_layers=12, num_heads=12, num_actions=11):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = PositionalEncoding(embed_dim, num_patches)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_actions)
        self.attention_pooling = AttentionPooling(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.encoder(x)
        x = self.attention_pooling(x)
        x = self.fc(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(val_loader)
    val_acc = 100 * correct / total
    return val_loss, val_acc

def NN_approx_init_states(all_obs, all_labels, num_epochs=100, confidence_threshold=0.51):
    X_train, X_test, y_train, y_test = train_test_split(all_obs, all_labels, test_size=0.2, random_state=42)

    transform_augment = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
    ])

    transform_original = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_dataset = CustomDataset(X_train, y_train, transform=transform_original)
    augmented_dataset = CustomDataset(X_train, y_train, transform=transform_augment)

    combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

    # Data loaders
    train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(CustomDataset(X_test, y_test, transform=transform_original), batch_size=32, shuffle=False)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    num_classes = len(np.unique(y_train))  # Adjust based on your dataset

    model = VisionTransformer(num_actions=num_classes)  # Adjust num_actions as per the unique classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    patience = 10
    counter = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

        # Check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0  # Reset counter if validation accuracy improves
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

        # Step the scheduler
        scheduler.step()

    # Final evaluation
    evaluate_model_with_confidence_and_exclusions(model, test_loader, device,
                                                  confidence_threshold=confidence_threshold, 
                                                  exclude_label=num_classes)

    print("Finished Training")
    
    

if __name__ == "__main__":
    
    args = tyro.cli(Args)
    
    if not args.incremental:
        remove_everything_in_folder(f"fracos_clusters/{args.env_id}")
    
    env_name = args.env_id

    saved_traj_dir = f"/home/x4nno/Documents/PhD/FRACOs_a/trajectories/e2e_traj/{args.env_id}/"
        
    traj_path = saved_traj_dir+"trajs.p"
    rew_path = saved_traj_dir+"rews.p"
    traj_content = pickle.load(open(traj_path, "rb"))
    rew_content = pickle.load(open(rew_path, "rb"))
    
    all_trajectories = traj_content
    all_ep_rewards = rew_content
    
    ######### remove ##############
    all_trajectories = all_trajectories[:10]
    all_ep_rewards = all_ep_rewards[:10]
    ##################################
    
    if args.rm_fail:
        success_idxs = [index for index, value in enumerate(all_ep_rewards) if value > args.failure_min]
        all_trajectories = [all_trajectories[idx] for idx in success_idxs]
        all_ep_rewards = [all_ep_rewards[idx] for idx in success_idxs]
        
    # Rather than what we have done here above, why don't we do the same clustering method and then approximate those states?
        
    
    fractures = create_fractures(all_trajectories, args.chain_length)
    fractures = [item for subfrac in fractures for item in subfrac]
    # get all our obs
    
    # not needed if we use rm_fail
    all_s_f = get_all_s_f_index(all_trajectories, all_ep_rewards, failure_std_threshold=None,
                                use_std=False, failure_min=args.failure_min)
        
    #### Cluster based on the action chains only ####
    grouped_obs = defaultdict(list)

    for entry in fractures:
        key = tuple([tuple(number) for number in entry[1:]])
        grouped_obs[key].append(entry[0])
        
        
    # Only take the top amount. ######## this needs to be our PI compression no?
    
    lengths = [(key, len(value)) for key, value in grouped_obs.items()]
    sorted_lengths = sorted(lengths, key=lambda item: item[1], reverse=True)
    
    
    labeled_keys = {i: key[0] for i, key in enumerate(sorted_lengths)}  # label maps to the key 
    
    reverse_labeled_keys = {v: k for k, v in labeled_keys.items()}   # key maps to the label
    
    top_x_keys = sorted_lengths[:args.max_clusters_per_clusterer]
    top_x_labels = [reverse_labeled_keys[key[0]] for key in top_x_keys]
    
    
    all_obs = []
    all_labels = []
    for key in grouped_obs.keys():
        for obs in grouped_obs[key]:
            all_obs.append(obs)
            if reverse_labeled_keys[key] in top_x_labels:
                all_labels.append(reverse_labeled_keys[key])
            else:
                all_labels.append(args.max_clusters_per_clusterer)
    
    
    
    #### Create a NN to predict the clusters that are available #### 
    
    init_model = NN_approx_init_states(all_obs, all_labels)
    
    print("debug point 2")
    
    #### Save the init_NN, the mapping and the dictionary
    
    
    # # need to come up with a new way of doing this too
    # # clusterer, top_cluster, all_success_clusters,\
    # #         ordered_cluster_pi_dict, best_clusters_list = \
    # #                 cluster_PI_compression(clusterer, images, all_s_f, all_trajectories,
    # #                                         chain_length=args.chain_length, max_cluster_returns=10000, 
    # #                                         min_PI_score = args.min_PI_score)
                    
    # save_all_clusterings(clusterer, best_clusters_list, images, 
    #                      all_trajectories, None, None, None, args.current_depth, env_name)
    
    # init_model = NN_approx_init_states(clusterer, best_clusters_list, features, images, class_type="multi", vae=model, num_epochs=args.NN_epochs)
    
    # os.makedirs(f"fracos_clusters/{args.env_id}/a/{args.current_depth}", exist_ok=True)
    
    # torch.save(init_model.state_dict(), f"fracos_clusters/{args.env_id}/a/{args.current_depth}/initiation.pth")
    # # if multi:
    # # pickle.dump([obs_shape, len(np.unique(clusterer.labels_)), best_clusters_list[:args.max_clusters_per_clusterer]], open(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/init_args.pkl", "wb"))
    # # if binary:
    # pickle.dump(50, open(f"fracos_clusters/{args.env_id}/a/{args.current_depth}/init_args.pkl", "wb"))