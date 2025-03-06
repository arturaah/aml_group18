# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
import pdb

class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, M, num_components=2):
        """
        Define a Mixture of Gaussians (MoG) prior with `num_components` Gaussians.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        num_components: [int]
           Number of Gaussian components in the mixture.
        """
        super(MixtureOfGaussiansPrior, self).__init__()
        self.M = M
        self.num_components = num_components

        # make parameters learnable
        self.means = nn.Parameter(torch.randn(num_components, M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(num_components, M), requires_grad=True)
        self.weights = nn.Parameter(torch.ones(num_components), requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Parameters:
        x: [torch.Tensor, optional]
           Input tensor (not used in this case).

        Returns:
        prior: [torch.distributions.Distribution]
        """
        
        mixture_distribution = td.Categorical((F.softmax(self.weights, dim=0)))
        component_distribution = td.Independent(td.Normal(self.means, self.stds), 1)
        mixture = td.MixtureSameFamily(mixture_distribution, component_distribution)
        return mixture


class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)
    
    
class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        mean = self.decoder_net(z)
        std = torch.ones_like(mean) * 0.1  # You can also make std a learnable parameter
        return td.Independent(td.Normal(mean, std), 1)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x) ## Posterior distribution
        #pdb.set_trace()
        z = q.rsample()
        
        ## ____ Old code ____
        
        # ## Gaussuian prior
        # if isinstance(self.prior, GaussianPrior):
        #     elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0)

        # ## Mixture of Gaussians prior (MoG) - no closed form solution, therefore we use MC estimate
        # elif isinstance(self.prior,MixtureOfGaussiansPrior):
        #     log_qz = q.log_prob(z)  # Log prob under q, posterior
        #     log_pz = self.prior().log_prob(z)  # Log prob under MoG prior
        #     kl_div = log_qz - log_pz  # Monte Carlo KL estimate
        #     elbo = torch.mean(self.decoder(z).log_prob(x) - kl_div, dim=0)
        
        ## ___ Above is old code _____
        
        
        if isinstance(self.prior, GaussianPrior):
            kl_div = td.kl_divergence(q, self.prior())
            
        elif isinstance(self.prior,MixtureOfGaussiansPrior):
            log_qz = q.log_prob(z)  # Log prob under q, posterior
            log_pz = self.prior().log_prob(z)  # Log prob under MoG prior
            kl_div = log_qz - log_pz  # Monte Carlo KL estimate
    
        if isinstance(self.decoder, GaussianDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x).sum(dim=1) - kl_div, dim=0)  
            
        elif isinstance(self.decoder, BernoulliDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x) - kl_div, dim=0)
            
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            #pdb.set_trace()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            

def eval(model, data_loader, device):
    """
    Evaluate a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo += model.elbo(x).sum().item() # Item gives a scalar
    avg_elbo = elbo/len(data_loader.dataset)
    print(f"The avg elbo is {avg_elbo:.2f}")
    
def plot_mog_prior_contour(prior_name, prior, device, grid_size=100, lim=5):
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
    plt.savefig(f'./plots/{prior_name}_prior_contour.png')

def plot_agg_posterior(model, data_loader, device):
    """
    Plot the aggregated posterior distribution.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    agg_z = []
    labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = model.encoder(x)
            z = q.rsample()
            agg_z.append(z.cpu())
            labels.append(y.cpu())
            
    agg_z = torch.cat(agg_z, dim=0)
    labels = torch.cat(labels, dim=0)
    #pdb.set_trace()
    # Perform PCA if latent dimension is greater than 2
    if agg_z.size(1) > 2:
        pca = PCA(n_components=2)
        agg_z_pca = pca.fit_transform(agg_z)
    else:
        agg_z_pca = agg_z.numpy()

    # Plot the samples from the approximate posterior
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(agg_z_pca[:, 0], agg_z_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Samples from the Approximate Posterior')
    plt.savefig('./plots/agg_posterior.png')
    #plt.show()
    
    
    
    

def plot_combined_prior_posterior(prior_name, prior, model, data_loader, device, grid_size=100, lim=5, n=5):
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

    plt.savefig(f'./plots/combined_{prior_name}_prior_posterior.png')
    #plt.show()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='MoG', choices=['MoG', 'Gaussian'], help='Define the prior (default: %(default)s)')
    parser.add_argument('--non-binarize', action='store_false', dest='binarize',
                    help="Use non-binarized MNIST. If not specified, binarized MNIST is used by default.")


    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    if args.binarize: 
        ## Load MNIST as binarized at 'thresshold' and create data loaders
        thresshold = 0.5
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
        
        
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
    
    else:
        ## Non-binarized MNIST
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.squeeze())
                ])))
        
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x.squeeze())
                ])))
    

    # Make sure plot dir and models dir exits
    if not os.path.exists("./plots"):
        os.makedirs("./plots")
            
    if not os.path.exists("./models"):
        os.makedirs("./models")
    
    # Define prior distribution
    M = args.latent_dim
    if args.prior == 'MoG':
        prior = MixtureOfGaussiansPrior(M, num_components=10)
    elif args.prior == 'Gaussian':
        prior = GaussianPrior(M)
    else:
        raise ValueError(f"Unknown prior type: {args.prior}")
        

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    if args.binarize:
        decoder = BernoulliDecoder(decoder_net)
    else:
        decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Save model
    model_name = f"{args.prior}_{args.model}"
    save_model_path = os.path.join("./models",model_name)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)
        
        torch.save(model.state_dict(), save_model_path)

    elif args.mode == 'sample':
        #pdb.set_trace()
        model.load_state_dict(torch.load(save_model_path, map_location=torch.device(args.device)))
        
        # Calculate avg elbo
        avg_elbo = eval(model, mnist_test_loader, device)
        
        # Plot aggregated posterior
        #plot_agg_posterior(model, mnist_test_loader, device)
        
        # Plot prior (mog)
        plot_mog_prior_contour(args.prior, prior, device,lim=10)
        
        plot_combined_prior_posterior(args.prior, prior, model, mnist_test_loader, device, lim=10)
        
        # Generate samples
        samples_name = f"{args.prior}_{args.samples}"
        save_samples_path = os.path.join("./plots",samples_name)
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), save_samples_path) ## .view reshapes the tensor to the desired shape
