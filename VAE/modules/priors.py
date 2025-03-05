import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data

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
    
class MoGPrior(nn.Module):
    def __init__(self, M, K=10):
        """
        Define a Mixture of Gaussians prior distribution with K components.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int] 
           Number of components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M # latent space
        self.K = K # number of components in the mixture

        # make parameters learnable
        self.means = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.K, self.M), requires_grad=True)
        self.weights = nn.Parameter(torch.ones(self.K), requires_grad=True)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # make weights sum to 1 and be non-negative using softmax
        weights = nn.functional.softmax(self.weights, dim=0)
        # make stds non-negative by taking the exponential
        stds = torch.exp(self.stds)
        # alternatively, we can use softplus to ensure that the standard deviations are positive
        # we also add a small constant to avoid 0 wich would make the likelihood undefined 
        #stds = torch.nn.functional.softplus(self.stds) + 1e-6

        mix = td.Categorical(probs=weights)
        comp = td.Independent(td.Normal(loc=self.means, scale=stds), 1)
        return td.MixtureSameFamily(mix, comp)
    
class VampPrior(nn.Module):
    def __init__(self, M, encoder, K=10, device='cpu'):
        """
        Define a VampPrior distribution with K pseudo-inputs.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        encoder: [nn.Module]
                    Encoder network.
        K: [int]
            Number of pseudo-inputs.       
        """
        super(VampPrior, self).__init__()
        self.M = M
        self.encoder = encoder
        self.K = K
        self.device = device
        # initialize the pseudo-input which are learnable
        # dimensions need to match the input dimensions of the encoder
        self.pseudo_inputs = nn.Parameter(torch.randn(self.K, 1, 28, 28), requires_grad=True)
        # initialize the mixing probabilities which are fixed and is just the average
        self.mixing_probs = nn.Parameter(torch.ones(self.K), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        # create a a mixture of the gaussian components provided by the encoder
        mix = td.Categorical(probs=self.mixing_probs)
        comp = self.encoder(self.pseudo_inputs)
        return td.MixtureSameFamily(mix, comp)

