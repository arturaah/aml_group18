import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)

        self.mse = nn.MSELoss()
    
    def network_pass(self, x, t):
        # pass x through the network but normalize t to be in range [0,1]
        t = t / self.T
        return self.network(x, t)

    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """
        ### Implement Algorithm 1 here ###              
        t = torch.randint(1, self.T+1, (x.shape[0], 1)).to(x.device) # sample t ~ U(1, T)        
        noise = torch.randn(x.shape).to(x.device) #sample noise ~ N(0, 1)
        # simplified loss function (eq 14) in the paper:
        noisy_img = torch.sqrt(self.alpha_cumprod[t-1]) * x + torch.sqrt(1 - self.alpha_cumprod[t-1]) * noise
        predicted_noise = self.network_pass(noisy_img, t)
        # mean squared error loss between the predicted noise and the actual noise
        neg_elbo = self.mse(noise,predicted_noise)
        return neg_elbo     

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            if t > 1:
                z = torch.randn(shape).to(self.alpha.device)
            else:
                z = 0
            # step 4 in Algorithm 2:
            # make t a tensor of shape (batch_size, 1) to match the shape of x_t
            t = torch.full((x_t.shape[0],1), t).to(x_t.device)
            x_t = (1 / torch.sqrt(self.alpha[t])
                    * (x_t - self.network_pass(x_t, t)* (1-self.alpha[t])/(torch.sqrt(1-self.alpha_cumprod[t]))
                    + torch.sqrt(self.beta[t]) * z))     
        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()