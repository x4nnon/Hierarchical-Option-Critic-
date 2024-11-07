import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import random
from dataclasses import dataclass
import tyro
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class Args:
    # Algorithm specific arguments
    env_id: str = "procgen-coinrun"
    """the id of the environment: MetaGridEnv/metagrid-v0, LunarLander-v2, procgen-coinrun, procgen-caveflyter,
    atari:BreakoutNoFrameskip-v4"""
    number_its: int = 50
    """How many epochs to run the vae learner"""
    show_ims: bool = False # make sure it false as default
    """If this is on, then it will print the recon and original image next to each other for comparison."""
    latent_dims: int = 20

class VAE(nn.Module):
    def __init__(self, channels, latent_dim, input_height, input_width):
        super(VAE, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        
        # Initial input dimensions
        self.input_height = input_height
        self.input_width = input_width
        
        ks, s, p = 2, 2, 1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=ks, stride=ks, padding=p),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=ks, stride=ks, padding=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=ks, stride=ks, padding=p),
            nn.ReLU()
            
            # nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU()
        )

        # Calculate the output dimensions after all convolutions
        self.conv_out_height = conv_output_size(conv_output_size(conv_output_size(input_height, ks, s, p), ks, s, p),ks, s, p)
        self.conv_out_width = conv_output_size(conv_output_size(conv_output_size(input_width, ks, s, p), ks, s, p), ks, s, p)
        conv_out_size = 64 * self.conv_out_height * self.conv_out_width

        # Fully connected layers for the latent space representation
        self.fc_mu = nn.Linear(conv_out_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_size, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, conv_out_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.conv_out_height, self.conv_out_width)),
            nn.ConvTranspose2d(64, 32, kernel_size=ks, stride=ks, padding=p, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=ks, stride=ks, padding=p, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=ks, stride=ks, padding=p, output_padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h.view(h.size(0), -1))
        logvar = self.fc_logvar(h.view(h.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    
    
class VAE_procgen(nn.Module):
    def __init__(self, channels, latent_dim, input_height, input_width):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        
        # Initial input dimensions
        self.input_height = input_height
        self.input_width = input_width
        
        ks, s, p = 2, 2, 1

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=ks, stride=ks, padding=p),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=ks, stride=ks, padding=p),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=ks, stride=ks, padding=p),
            nn.ReLU()
            
            # nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU()
        )

        # Calculate the output dimensions after all convolutions
        self.conv_out_height = conv_output_size(conv_output_size(conv_output_size(input_height, ks, s, p), ks, s, p),ks, s, p)
        self.conv_out_width = conv_output_size(conv_output_size(conv_output_size(input_width, ks, s, p), ks, s, p), ks, s, p)
        conv_out_size = 64 * self.conv_out_height * self.conv_out_width

        # Fully connected layers for the latent space representation
        self.fc_mu = nn.Linear(conv_out_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_size, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, conv_out_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, self.conv_out_height, self.conv_out_width)),
            nn.ConvTranspose2d(64, 32, kernel_size=ks, stride=ks, padding=p, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=ks, stride=ks, padding=p, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, kernel_size=ks, stride=ks, padding=p, output_padding=0),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return torch.sigmoid(z)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h.view(h.size(0), -1))
        logvar = self.fc_logvar(h.view(h.size(0), -1))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    
def conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1    


def vae_loss(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def create_dataloader(data, batch_size):
    tensor_data = torch.from_numpy(data).float()
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def show_recon(recon, data):
    for i in range(5):
        recon_image = recon[i].permute(1,2,0).cpu().numpy()
        data_image = data[i].permute(1,2,0).cpu().numpy()
    
        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot the reconstructed image
        axes[0].imshow(recon_image, cmap='gray')
        axes[0].set_title('Reconstructed Image')
        axes[0].axis('off')  # Hide axis
        
        # Plot the original data image
        axes[1].imshow(data_image, cmap='gray')
        axes[1].set_title('Original Image')
        axes[1].axis('off')  # Hide axis
        
        # Display the plots
        plt.show()

    
def train_VAE(e2e_traj_dir, args):
    with open(e2e_traj_dir, "rb") as f:
        e2e_trajs = pickle.load(f)
        
    epochs = args.number_its
    show_ims = args.show_ims
    
    all_Os = []
    for traj in e2e_trajs:
        for O, A in traj:
            all_Os.append(O)
    
    all_Os = np.array(all_Os)
    if "procgen" in args.env_id:
        all_Os = all_Os.transpose(0, 3, 1, 2)
    
    
    all_Os = all_Os.astype(np.float32) / 255.0
    
    train_images, test_images = train_test_split(all_Os, test_size=0.2, random_state=42)  # Split data
    
    
    
    batch_size = 32
    train_loader = create_dataloader(train_images, batch_size)
    test_loader = create_dataloader(test_images, batch_size)
    
    # Convert to PyTorch tensor and reshape to (num_samples, channels, height, width)
    
    # Create a TensorDataset and DataLoader

    
    
    channels = all_Os.shape[1]  
    latent_dims = args.latent_dims
    input_height = all_Os.shape[-1]
    input_width = all_Os.shape[-1]
    
    if "atari" in args.env_id:
        model = VAE(channels, latent_dims, input_height, input_width).to(device)
    elif "procgen" in args.env_id:
        model = VAE_procgen(channels, latent_dims, input_height, input_width).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f'====> Epoch: {epoch} Average training loss: {train_loss / len(train_loader.dataset)}')
    
        model.eval()
        test_loss = 0
        first = True
        with torch.no_grad():
            for batch_idx, (data,) in enumerate(test_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                
                if first and show_ims:
                    if random.random() < 0.1:
                        show_recon(recon, data)
                        first=False
                    
                test_loss += vae_loss(recon, data, mu, logvar).item()
    
        test_loss /= len(test_loader.dataset)
        print(f'====> Epoch: {epoch} Average test loss: {test_loss}')
        
    directory_path = f"vae_models/{args.env_id}"
    os.makedirs(directory_path, exist_ok=True)
    torch.save(model.state_dict(), f"{directory_path}/model.pth")
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    args.show_ims = True
    traj_dir = f"trajectories/e2e_traj/{args.env_id}/trajs.p"
    train_VAE(traj_dir, args)
