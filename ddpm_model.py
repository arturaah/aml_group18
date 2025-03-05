import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.embedder = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, t):
        # Takes a tensor of timesteps and returns their embeddings
        t_reshape = t.reshape(-1, 1).float()
        return self.embedder(t_reshape)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        
        # Optional time embedding projection
        self.time_mlp = None
        if time_dim is not None:
            self.time_mlp = nn.Linear(time_dim, out_ch)

    def forward(self, x, t=None):
        h = self.conv1(x)
        h = self.norm(h)
        
        # Add time embedding if provided
        if self.time_mlp is not None and t is not None:
            time_emb = self.time_mlp(t)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
            
        h = self.act(h)
        return h

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t=None):
        skip = self.conv(x, t)
        x = self.pool(skip)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + out_ch, out_ch, time_dim)

    def forward(self, x, skip, t=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Encoder (downsampling)
        self.inc = ConvBlock(in_channels, 64, time_dim)
        self.down1 = DownBlock(64, 128, time_dim)
        self.down2 = DownBlock(128, 256, time_dim)
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512, time_dim)
        
        # Decoder (upsampling)
        self.up1 = UpBlock(512, 256, time_dim)
        self.up2 = UpBlock(256, 128, time_dim)
        self.outc = nn.Sequential(
            ConvBlock(128, 64, time_dim),
            nn.Conv2d(64, out_channels, 1)
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Encoder
        x1 = self.inc(x, t_emb)
        x2, skip1 = self.down1(x1, t_emb)
        x3, skip2 = self.down2(x2, t_emb)
        
        # Bottleneck
        x4 = self.bottleneck(x3, t_emb)
        
        # Decoder
        x = self.up1(x4, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        x = self.outc[0](x, t_emb)
        x = self.outc[1](x)
        
        return x

class DDPM(nn.Module):
    def __init__(self, model, beta_start=1e-4, beta_end=0.02, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # Define beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        
        # Pre-calculate diffusion process parameters
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers - this ensures they're moved to the right device with the model
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def forward(self, x):
        """Training forward pass"""
        batch_size = x.shape[0]
        device = x.device
        
        # Sample t uniformly
        t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        epsilon = torch.randn_like(x)
        
        # Forward diffusion process: get x_t
        x_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x + \
              self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * epsilon
        
        # Fix: This line doesn't do anything as it returns a new tensor without assignment
        # x_t.to(device)
        # The tensor is already on the correct device so this line can be removed
        
        # Let the model predict the noise
        predicted_noise = self.model(x_t, t)
        
        return F.mse_loss(predicted_noise, epsilon)

    def sample(self, num_samples, img_shape, device):
        """Generate samples from the trained model"""
        model = self.model
        model.eval()
        
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(num_samples, *img_shape, device=device)
            
            # Iterative denoising
            for t in range(self.timesteps - 1, -1, -1):
                t_batch = torch.ones(num_samples, device=device).long() * t
                
                # Predict noise
                predicted_noise = model(x, t_batch)
                
                # Extract parameters for the current timestep
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                # No noise on the last step
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                    
                # Equation 12 from the DDPM paper (simplified for direct sampling)
                x = 1 / torch.sqrt(alpha) * (
                    x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise
                ) + torch.sqrt(beta) * noise
                
        # Scale to [0, 1] for visualization
        return torch.clamp(x, -1, 1) * 0.5 + 0.5