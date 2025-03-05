import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_combined_prior_posterior(plot_name, prior, model, data_loader, device, grid_size=100, lim=5, n=5):
    """
    Plots the contour of the prior distribution and overlays the aggregated posterior samples.

    Parameters:
    - prior: MixtureOfGaussiansPrior instance
    - model: VAE model to evaluate
    - data_loader: DataLoader providing input data
    - device: torch device ('cpu' or 'cuda')
    - grid_size: Number of points per axis in the grid
    - lim: Limits for the plot in latent space
    """
        
    prior = prior.to(device)
    model.eval()
    
    # Generate 2D grid for contour plot
    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x, y)
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    grid_torch = torch.tensor(grid, dtype=torch.float32, device=device)
    
    # If latent space is high-dimensional, perform PCA
    if prior.M > 2:
        pca = PCA(n_components=2)
        
        # Sample from prior to determine PCA projection
        z_samples = prior().sample((1000,)).detach().cpu().numpy()
        pca.fit(z_samples)
        
        # Transform the grid points
        grid_torch = torch.tensor(pca.inverse_transform(grid), dtype=torch.float32, device=device)
    else:
        pca = None  # No need for PCA

    # Compute log probability of prior
    log_probs = prior().log_prob(grid_torch).detach().cpu().numpy()
    log_probs = log_probs.reshape(grid_size, grid_size)

    # Plot prior contour
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, log_probs, levels=30, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, label='Log Probability')

    # Compute aggregated posterior samples
    agg_z, labels = [], []
    with torch.no_grad():
        #for x, y in data_loader:
        for i, (x, y) in enumerate(data_loader):
            if i % n == 0:
                x = x.to(device)
                q = model.encoder(x)
                z = q.rsample()
                agg_z.append(z.cpu())
                labels.append(y.cpu())

    agg_z = torch.cat(agg_z, dim=0)
    labels = torch.cat(labels, dim=0)

    # Apply PCA if needed
    if agg_z.size(1) > 2 and pca is not None:
        agg_z_pca = pca.transform(agg_z)
    else:
        agg_z_pca = agg_z.numpy()

    # Scatter plot of posterior samples
    scatter = ax.scatter(agg_z_pca[:, 0], agg_z_pca[:, 1], c=labels, cmap='tab10', alpha=0.7, edgecolors='k')
    plt.colorbar(scatter, label='Class Label')

    # Labels and title
    ax.set_xlabel("Latent Dimension 1" if pca is None else "Principal Component 1")
    ax.set_ylabel("Latent Dimension 2" if pca is None else "Principal Component 2")
    ax.set_title("Prior Contour with Aggregated Posterior Samples")

    plt.savefig(plot_name)
    
def plot_mog_prior_contour(plot_name, prior, device, grid_size=100, lim=5):
    """
    Plot the contour of a Mixture of Gaussians (MoG) prior.
    
    Parameters:
    - prior: MixtureOfGaussiansPrior or Gaussian instance
    - device: torch device ('cpu' or 'cuda')
    - grid_size: Number of points per axis in the grid
    - lim: Limits for the plot in latent space
    """
    prior = prior.to(device)
    
    # Generate 2D grid
    x = np.linspace(-lim, lim, grid_size)
    y = np.linspace(-lim, lim, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the grid for evaluation
    grid = np.vstack([X.ravel(), Y.ravel()]).T
    grid_torch = torch.tensor(grid, dtype=torch.float32, device=device)
    
    if hasattr(prior, 'M') and prior.M > 2:
        # Use PCA to project the high-dimensional latent space to 2D
        pca = PCA(n_components=2)
        z_samples = prior().sample((1000,)).detach().numpy()  # Sample from MoG
        pca.fit(z_samples)
        grid_torch = torch.tensor(pca.inverse_transform(grid), dtype=torch.float32, device=device)

    # Compute log probability from the prior
    log_probs = prior().log_prob(grid_torch).detach().numpy()
    log_probs = log_probs.reshape(grid_size, grid_size)  # Reshape for contour plot

    # Plot contour
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, log_probs, levels=30, cmap='viridis')
    plt.colorbar(contour, label='Log Probability')
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title("Contour Plot of Prior")
    plt.savefig(plot_name)