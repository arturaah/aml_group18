# Sampling quality of generative models

import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import for F.binary_cross_entropy
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import os

# Import our model implementations
from flow_model import FlowModel
from ddpm_model import DDPM, UNet
from vae_model import VAE

# Import the metrics tracking utilities
from metrics_utils import MetricsTracker, compare_generation_quality, visualize_latent_space

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create models directory
os.makedirs("models", exist_ok=True)
os.makedirs("samples", exist_ok=True)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
])

# For VAE comparison on binarized MNIST
transform_binarized = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.bernoulli(x))  # Binarize the data
])

# Load standard (non-binarized) MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False,
    download=True, 
    transform=transform
)

# Load binarized MNIST for VAE comparison
train_dataset_binarized = torchvision.datasets.MNIST(
    root='./data', 
    train=True,
    download=True, 
    transform=transform_binarized
)

# Use smaller dataset for faster training during development
train_subset = Subset(train_dataset, range(10000))
train_subset_binarized = Subset(train_dataset_binarized, range(10000))

batch_size = 128
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
train_loader_binarized = DataLoader(train_subset_binarized, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Helper function to visualize images
def show_images(images, title="", save_path=None):
    plt.figure(figsize=(8, 8))
    for i, img in enumerate(images):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if len(img.shape) == 3:
            img = img.transpose(1, 2, 0)  # CxHxW -> HxWxC
        plt.subplot(2, 2, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    
    # Always save to working directory if no path specified
    if save_path is None:
        save_path = f"samples/{title.replace(' ', '_').lower()}.png"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path)
    
    # Close the plot to prevent memory leaks
    plt.close()

# Helper function to save model
def save_model(model, model_name):
    torch.save(model.state_dict(), f"models/{model_name}.pth")

# Helper function to load model
def load_model(model, model_name):
    model.load_state_dict(torch.load(f"models/{model_name}.pth", map_location=device))
    return model

# Initialize metrics tracker
metrics_tracker = MetricsTracker()

# Train Flow model
def train_flow_model(epochs=50):
    # Initialize model
    flow_model = FlowModel(in_channels=1, hidden_dim=64, num_blocks=4).to(device)
    optimizer = optim.Adam(flow_model.parameters(), lr=1e-3)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        flow_model.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            _, loss = flow_model(data)
            loss = loss.mean()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
                
        # Calculate average loss for this epoch
        avg_loss = train_loss/len(train_loader)
        epoch_losses.append(avg_loss)
        
        # Track loss with metrics tracker
        metrics_tracker.add_metric('flow_losses', avg_loss)
                
        # Print epoch stats
        print(f'Epoch: {epoch+1}, Average loss: {avg_loss:.4f}')
        
        # Generate and save samples 
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_flow_samples(flow_model, 4, f"samples/flow_epoch_{epoch+1}.png")
    
    # Save the model
    save_model(flow_model, "flow_model")
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, 'r-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Flow Model Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('samples/flow_loss.png')
    plt.close()
    
    return flow_model

# Train DDPM model
def train_ddpm_model(epochs=10):
    # Initialize models
    unet = UNet(in_channels=1, out_channels=1, time_dim=256).to(device)
    ddpm = DDPM(unet, timesteps=1000).to(device)
    optimizer = optim.Adam(ddpm.parameters(), lr=1e-4)
    
    epoch_losses = []
    
    for epoch in range(epochs):
        ddpm.train()
        train_loss = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            # Scale to [-1, 1] for better diffusion stability
            data = data * 2 - 1
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = ddpm(data)
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average loss for this epoch
        avg_loss = train_loss/len(train_loader)
        epoch_losses.append(avg_loss)
        
        # Track loss with metrics tracker
        metrics_tracker.add_metric('ddpm_losses', avg_loss)
                
        # Print epoch stats
        print(f'Epoch: {epoch+1}, Average loss: {avg_loss:.4f}')
        
        # Generate and save samples
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            generate_ddpm_samples(ddpm, 4, f"samples/ddpm_epoch_{epoch+1}.png")
    
    # Save the model
    save_model(ddpm, "ddpm_model")
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, 'g-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('DDPM Model Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('samples/ddpm_loss.png')
    plt.close()
    
    return ddpm

# Train VAE model
def train_vae_model(epochs=10, binarized=False):
    # Initialize model
    vae = VAE(latent_dim=20).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    # Choose dataset based on binarized flag
    data_loader = train_loader_binarized if binarized else train_loader
    model_name = "vae_model_binarized" if binarized else "vae_model"
    
    epoch_losses = []
    epoch_recon_losses = []
    epoch_kl_losses = []
    
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = vae(data)
            
            # Loss calculation
            recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_div
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_div.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average loss for this epoch
        avg_loss = train_loss/len(data_loader)
        avg_recon_loss = recon_loss_total/len(data_loader)
        avg_kl_loss = kl_loss_total/len(data_loader)
        
        epoch_losses.append(avg_loss)
        epoch_recon_losses.append(avg_recon_loss)
        epoch_kl_losses.append(avg_kl_loss)
        
        # Track loss with metrics tracker
        if binarized:
            metrics_tracker.add_metric('vae_binarized_losses', avg_loss)
        else:
            metrics_tracker.add_metric('vae_losses', avg_loss)
        
        metrics_tracker.add_metric('vae_recon_losses', avg_recon_loss)
        metrics_tracker.add_metric('vae_kl_losses', avg_kl_loss)
                
        # Print epoch stats
        print(f'Epoch: {epoch+1}, Avg Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')
        
        # Generate and save samples
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            bin_text = "binarized_" if binarized else ""
            generate_vae_samples(vae, 4, f"samples/vae_{bin_text}epoch_{epoch+1}.png")
    
    # Save the model
    save_model(vae, model_name)
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), epoch_losses, 'b-', label='Total Loss')
    plt.plot(range(1, epochs+1), epoch_recon_losses, 'g-', label='Reconstruction Loss')
    plt.plot(range(1, epochs+1), epoch_kl_losses, 'r-', label='KL Divergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'VAE Model Training Loss {"(Binarized)" if binarized else ""}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'samples/vae_loss{"_binarized" if binarized else ""}.png')
    plt.close()
    
    return vae

# Generate samples from Flow model
def generate_flow_samples(model, num_samples=4, save_path=None):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, img_shape=(1, 28, 28), device=device)
    
    # Display samples
    show_images(samples, "Flow Model Samples", save_path)
    return samples

# Generate samples from DDPM model
def generate_ddpm_samples(model, num_samples=4, save_path=None):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, img_shape=(1, 28, 28), device=device)
    
    # Display samples
    show_images(samples, "DDPM Model Samples", save_path)
    return samples

# Generate samples from VAE model
def generate_vae_samples(model, num_samples=4, save_path=None, binarized=False):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device=device)
    
    # Display samples
    title = "VAE Model Samples (Binarized)" if binarized else "VAE Model Samples"
    show_images(samples, title, save_path)
    return samples

# Main execution
if __name__ == "__main__":
    # Check if models exist, otherwise train them
    train_models = True
    
    if train_models:
        # Train all models
        print("Training Flow Model...")
        flow_model = train_flow_model(epochs=5)
        
        print("Training DDPM Model...")
        ddpm_model = train_ddpm_model(epochs=5)
        
        print("Training VAE Model on standard MNIST...")
        vae_model = train_vae_model(epochs=5, binarized=False)
        
        print("Training VAE Model on binarized MNIST...")
        vae_model_binarized = train_vae_model(epochs=5, binarized=True)
        
        # Plot combined loss evolution
        metrics_tracker.plot_losses()
        
        # Compare generation quality
        print("Comparing generation quality...")
        quality_metrics = compare_generation_quality(
            flow_model, ddpm_model, vae_model, vae_model_binarized,
            test_loader, device
        )
        print("FID Scores:")
        for model_name, score in quality_metrics.items():
            print(f"{model_name}: {score:.4f}")
            
        # Visualize VAE latent space
        print("Visualizing VAE latent space...")
        visualize_latent_space(vae_model, test_loader, device)
        
    else:
        # Load pre-trained models
        print("Loading pre-trained models...")
        flow_model = load_model(FlowModel(in_channels=1, hidden_dim=64, num_blocks=4).to(device), "flow_model")
        
        unet = UNet(in_channels=1, out_channels=1, time_dim=256).to(device)
        ddpm_model = load_model(DDPM(unet, timesteps=1000), "ddpm_model")
        
        vae_model = load_model(VAE(latent_dim=20).to(device), "vae_model")
        vae_model_binarized = load_model(VAE(latent_dim=20).to(device), "vae_model_binarized")
        
        # Compare generation quality for pre-trained models
        print("Comparing generation quality of pre-trained models...")
        quality_metrics = compare_generation_quality(
            flow_model, ddpm_model, vae_model, vae_model_binarized,
            test_loader, device
        )
        print("FID Scores:")
        for model_name, score in quality_metrics.items():
            print(f"{model_name}: {score:.4f}")
            
        # Visualize VAE latent space
        print("Visualizing VAE latent space...")
        visualize_latent_space(vae_model, test_loader, device)
    
    # Generate final samples
    print("Generating samples from all models...")
    flow_samples = generate_flow_samples(flow_model, 4, "samples/flow_final.png")
    ddpm_samples = generate_ddpm_samples(ddpm_model, 4, "samples/ddpm_final.png")
    vae_samples = generate_vae_samples(vae_model, 4, "samples/vae_final.png")
    vae_binarized_samples = generate_vae_samples(vae_model_binarized, 4, "samples/vae_binarized_final.png", binarized=True)