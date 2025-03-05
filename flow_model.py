import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels // 2, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1)
        )
        
        # Initialize last layer with zeros for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        
    def forward(self, x, reverse=False):
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        h = self.net(x1)
        s, t = torch.chunk(h, 2, dim=1)
        s = torch.sigmoid(s + 2)  # Output is bounded and initialized near 1
        
        if not reverse:
            y2 = x2 * s + t
            log_det = torch.sum(torch.log(s), dim=[1, 2, 3])
        else:
            y2 = (x2 - t) / s
            log_det = -torch.sum(torch.log(s), dim=[1, 2, 3])
            
        y = torch.cat([x1, y2], dim=1)
        return y, log_det

class Squeeze(nn.Module):
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        self.factor = factor
        
    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        if not reverse:
            # Squeeze: [B, C, H, W] -> [B, C*factor^2, H/factor, W/factor]
            x = x.view(B, C, H // self.factor, self.factor, W // self.factor, self.factor)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, C * self.factor * self.factor, H // self.factor, W // self.factor)
            return x, 0
        else:
            # Unsqueeze: [B, C*factor^2, H/factor, W/factor] -> [B, C, H, W]
            x = x.view(B, C // (self.factor * self.factor), self.factor, self.factor, 
                       H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(B, C // (self.factor * self.factor), H * self.factor, 
                      W * self.factor)
            return x, 0

class FlowModel(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64, num_blocks=4):
        super(FlowModel, self).__init__()
        self.flows = nn.ModuleList()
        
        # Layer structure: [Squeeze -> AffineCoupling] x num_blocks
        self.flows.append(Squeeze())
        in_ch = in_channels * 4  # After squeezing
        
        for _ in range(num_blocks):
            self.flows.append(AffineCoupling(in_ch, hidden_dim))
        
        # Learnable prior parameters (mean and log_std)
        self.prior_mean = nn.Parameter(torch.zeros(1))
        self.prior_log_std = nn.Parameter(torch.zeros(1))
            
    def forward(self, x, reverse=False):
        log_det_total = 0
        
        if not reverse:
            # Forward pass (encoding)
            for flow in self.flows:
                x, log_det = flow(x, reverse=False)
                log_det_total += log_det
                
            # Calculate NLL loss
            z = x
            log_p = -0.5 * torch.sum(
                ((z - self.prior_mean) / torch.exp(self.prior_log_std)) ** 2 + 
                2 * self.prior_log_std + np.log(2 * np.pi), 
                dim=[1, 2, 3])
            
            # Negative log-likelihood
            nll = -log_p - log_det_total
            return z, nll
            
        else:
            # Reverse pass (decoding / sampling)
            # Sample from prior
            z = torch.randn_like(x) * torch.exp(self.prior_log_std) + self.prior_mean
            
            # Apply flows in reverse order
            for flow in reversed(self.flows):
                z, _ = flow(z, reverse=True)
                
            return z
    
    def sample(self, num_samples, img_shape=(1, 28, 28), device='cpu'):
        # Create random noise
        z = torch.randn(num_samples, img_shape[0]*4, img_shape[1]//2, img_shape[2]//2, 
                        device=device) * torch.exp(self.prior_log_std) + self.prior_mean
        
        # Pass through reverse flows
        for flow in reversed(self.flows):
            z, _ = flow(z, reverse=True)
            
        # Make sure output is in [0, 1] range
        return torch.sigmoid(z)