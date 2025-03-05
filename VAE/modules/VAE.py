import torch
import torch.nn as nn
import torch.distributions as td
from VAE.modules.decoders import BernoulliDecoder
from VAE.modules.decoders import GaussianDecoder
from VAE.modules.priors import GaussianPrior, MoGPrior, VampPrior

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
        q = self.encoder(x)
        z = q.rsample()       
        
        if isinstance(self.prior, GaussianPrior):
            kl_div = td.kl_divergence(q, self.prior())
            
        elif isinstance(self.prior,MoGPrior) or isinstance(self.prior,VampPrior):
            log_qz = q.log_prob(z)  # Log prob under q, posterior
            log_pz = self.prior().log_prob(z)  # Log prob under MoG prior
            kl_div = log_qz - log_pz  # Monte Carlo KL estimate
    
        if isinstance(self.decoder, BernoulliDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x) - kl_div, dim=0)

        elif isinstance(self.decoder, GaussianDecoder):
            elbo = torch.mean(self.decoder(z).log_prob(x).sum(dim=1) - kl_div, dim=0)              
            
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