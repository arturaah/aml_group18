# Python file for project1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 100
learning_rate = 1e-3
epochs = 50
latent_dim = 40
hidden_dim = 300
n_components = 10  # For MoG prior
k_pseudo = 500     # Number of pseudo-inputs for VampPrior

# Data loading and binarization
def binarize(x):
    return (x > 0.5).float()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Base VAE model
class VAE(nn.Module):
    def __init__(self, prior_type='gaussian'):
        super(VAE, self).__init__()
        self.prior_type = prior_type
        
        # Encoder
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 784)
        
        # MoG prior parameters (if needed)
        if prior_type == 'mog':
            # Component means
            self.mog_means = nn.Parameter(torch.randn(n_components, latent_dim))
            # Component log variances
            self.mog_log_vars = nn.Parameter(torch.zeros(n_components, latent_dim))
            # Component weights (unnormalized)
            self.mog_weights = nn.Parameter(torch.ones(n_components))
        
        # VampPrior parameters (if needed)
        elif prior_type == 'vampprior':
            # Pseudo-inputs (in an idle form)
            self.idle_input = nn.Parameter(torch.zeros(k_pseudo, 784))
            # Pseudo-input embedding
            self.pseudo_means = nn.Parameter(0.05 * torch.randn(k_pseudo, latent_dim))
            self.idle_log_var = nn.Parameter(torch.zeros(k_pseudo, latent_dim))
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        log_var = self.fc22(h1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        
        # Encode
        mu, log_var = self.encode(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decode(z)
        
        return x_recon, mu, log_var, z
    
    def compute_prior_log_prob(self, z):
        if self.prior_type == 'gaussian':
            # Standard Gaussian prior
            return torch.sum(-0.5 * (z**2 + np.log(2 * np.pi)), dim=1)
        
        elif self.prior_type == 'mog':
            # Mixture of Gaussians prior
            z_expanded = z.unsqueeze(1)  # [B, 1, latent_dim]
            means = self.mog_means.unsqueeze(0)  # [1, n_components, latent_dim]
            log_vars = self.mog_log_vars.unsqueeze(0)  # [1, n_components, latent_dim]
            
            # Compute log prob for each component
            log_p_z_components = -0.5 * torch.sum(
                (z_expanded - means)**2 / torch.exp(log_vars) + log_vars + np.log(2 * np.pi),
                dim=2
            )  # [B, n_components]
            
            # Normalize weights
            log_weights = F.log_softmax(self.mog_weights, dim=0)  # [n_components]
            
            # Combine using log-sum-exp trick
            max_log_p_z = torch.max(log_p_z_components, dim=1, keepdim=True)[0]
            log_p_z = max_log_p_z + torch.log(
                torch.sum(
                    torch.exp(log_p_z_components - max_log_p_z) * torch.exp(log_weights).unsqueeze(0),
                    dim=1
                )
            )
            return log_p_z.squeeze()
        
        elif self.prior_type == 'vampprior':
            # VampPrior
            # Generate pseudo-inputs
            pseudo_inputs = torch.sigmoid(self.idle_input)
            
            # Encode pseudo-inputs
            pseudo_mu = self.pseudo_means
            pseudo_log_var = self.idle_log_var
            
            # Compute log prob for each pseudo-input
            z_expanded = z.unsqueeze(1)  # [B, 1, latent_dim]
            means = pseudo_mu.unsqueeze(0)  # [1, k_pseudo, latent_dim]
            log_vars = pseudo_log_var.unsqueeze(0)  # [1, k_pseudo, latent_dim]
            
            # Compute log prob for each component
            log_p_z_components = -0.5 * torch.sum(
                (z_expanded - means)**2 / torch.exp(log_vars) + log_vars + np.log(2 * np.pi),
                dim=2
            )  # [B, k_pseudo]
            
            # Uniform weights (1/K for each component)
            log_p_z = torch.logsumexp(log_p_z_components, dim=1) - torch.log(torch.tensor(k_pseudo, dtype=torch.float))
            
            return log_p_z
    
    def loss_function(self, x, x_recon, mu, log_var, z):
        # Reconstruction loss (Bernoulli)
        BCE = F.binary_cross_entropy(x_recon, x.view(-1, 784), reduction='sum')
        
        # Prior log probability
        log_p_z = self.compute_prior_log_prob(z)
        
        # KL divergence
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        
        # ELBO
        ELBO = torch.mean(log_p_z - KLD - BCE / batch_size)
        
        return -ELBO  # Negate ELBO for minimization
    
    def sample(self, n_samples=64):
        with torch.no_grad():
            if self.prior_type == 'gaussian':
                # Sample from standard Gaussian
                z = torch.randn(n_samples, latent_dim).to(device)
            
            elif self.prior_type == 'mog':
                # Sample from MoG
                # First sample component indices
                weights = F.softmax(self.mog_weights, dim=0)
                component_idx = torch.multinomial(weights, n_samples, replacement=True)
                
                # Then sample from selected Gaussians
                z = torch.zeros(n_samples, latent_dim).to(device)
                for i in range(n_samples):
                    idx = component_idx[i]
                    mu = self.mog_means[idx]
                    log_var = self.mog_log_vars[idx]
                    std = torch.exp(0.5 * log_var)
                    z[i] = mu + torch.randn_like(std) * std
            
            elif self.prior_type == 'vampprior':
                # Sample from VampPrior
                # First sample pseudo-input indices
                idx = torch.randint(0, k_pseudo, (n_samples,))
                
                # Then sample from corresponding Gaussians
                z = torch.zeros(n_samples, latent_dim).to(device)
                for i in range(n_samples):
                    mu = self.pseudo_means[idx[i]]
                    log_var = self.idle_log_var[idx[i]]
                    std = torch.exp(0.5 * log_var)
                    z[i] = mu + torch.randn_like(std) * std
            
            # Decode
            return self.decode(z)


# Training function
def train(model, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        x_recon, mu, log_var, z = model(data)
        loss = model.loss_function(data, x_recon, mu, log_var, z)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch: {epoch}, Train loss: {train_loss / len(train_loader.dataset):.4f}')
    return train_loss / len(train_loader.dataset)


# Test function
def test(model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, mu, log_var, z = model(data)
            loss = model.loss_function(data, x_recon, mu, log_var, z)
            test_loss += loss.item()
    
    test_loss /= len(test_loader.dataset)
    print(f'Test loss: {test_loss:.4f}')
    return test_loss


# Visualization functions
def visualize_reconstructions(model, n=10):
    model.eval()
    with torch.no_grad():
        # Get some test data
        data, _ = next(iter(test_loader))
        data = data[:n].to(device)
        
        # Reconstruct
        x_recon, _, _, _ = model(data)
        
        # Plot
        fig, axs = plt.subplots(2, n, figsize=(12, 4))
        for i in range(n):
            # Original images
            axs[0, i].imshow(data[i].cpu().view(28, 28), cmap='gray')
            axs[0, i].axis('off')
            if i == 0:
                axs[0, i].set_title('Original')
            
            # Reconstructed images
            axs[1, i].imshow(x_recon[i].cpu().view(28, 28), cmap='gray')
            axs[1, i].axis('off')
            if i == 0:
                axs[1, i].set_title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(f'reconstructions_{model.prior_type}.png')
        plt.show()


def visualize_prior_vs_posterior(model):
    model.eval()
    prior_samples = []
    posterior_samples = []
    
    # Collect samples from the prior
    with torch.no_grad():
        if model.prior_type == 'gaussian':
            prior_samples = torch.randn(1000, latent_dim).to(device)
        elif model.prior_type == 'mog':
            weights = F.softmax(model.mog_weights, dim=0)
            component_idx = torch.multinomial(weights, 1000, replacement=True)
            for i in range(1000):
                idx = component_idx[i]
                mu = model.mog_means[idx]
                log_var = model.mog_log_vars[idx]
                std = torch.exp(0.5 * log_var)
                prior_samples.append(mu + torch.randn_like(std) * std)
            prior_samples = torch.stack(prior_samples)
        elif model.prior_type == 'vampprior':
            idx = torch.randint(0, k_pseudo, (1000,))
            for i in range(1000):
                mu = model.pseudo_means[idx[i]]
                log_var = model.idle_log_var[idx[i]]
                std = torch.exp(0.5 * log_var)
                prior_samples.append(mu + torch.randn_like(std) * std)
            prior_samples = torch.stack(prior_samples)
    
    # Collect samples from the aggregate posterior
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            data = data.to(device)
            mu, log_var = model.encode(data.view(-1, 784))
            z = model.reparameterize(mu, log_var)
            posterior_samples.append(z)
    
    posterior_samples = torch.cat(posterior_samples)
    
    # Select 2 dimensions to visualize
    dim1, dim2 = 0, 1
    
    plt.figure(figsize=(10, 6))
    plt.scatter(prior_samples[:1000, dim1].cpu(), prior_samples[:1000, dim2].cpu(), 
                alpha=0.5, label='Prior', color='blue')
    plt.scatter(posterior_samples[:1000, dim1].cpu(), posterior_samples[:1000, dim2].cpu(), 
                alpha=0.5, label='Aggregate Posterior', color='red')
    plt.xlabel(f'Dimension {dim1}')
    plt.ylabel(f'Dimension {dim2}')
    plt.title(f'Prior vs Aggregate Posterior ({model.prior_type})')
    plt.legend()
    plt.savefig(f'prior_vs_posterior_{model.prior_type}.png')
    plt.show()


# Main function to train VAEs with different priors
def run_experiment():
    prior_types = ['gaussian', 'mog', 'vampprior']
    results = {}
    
    for prior_type in prior_types:
        print(f"\n\n--- Training VAE with {prior_type} prior ---")
        model = VAE(prior_type=prior_type).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        for epoch in range(1, epochs + 1):
            train_loss = train(model, optimizer, epoch)
            train_losses.append(train_loss)
            if epoch % 10 == 0:
                test_loss = test(model)
        
        # Final evaluation
        test_loss = test(model)
        results[prior_type] = test_loss
        
        # Visualizations
        visualize_reconstructions(model)
        visualize_prior_vs_posterior(model)
        
        # Save model
        torch.save(model.state_dict(), f'vae_{prior_type}.pt')
    
    # Compare results
    print("\n--- Test Log Likelihood (ELBO) Comparison ---")
    for prior_type, elbo in results.items():
        print(f"{prior_type}: {-elbo:.4f}")  # Negate because we minimized -ELBO


if __name__ == "__main__":
    run_experiment()
