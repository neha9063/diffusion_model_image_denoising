"""
Diffusion Model for Image Denoising
Core Neural Network Architecture & Diffusion Math Implementation
Compatible with MNIST (28x28 grayscale images)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


# ============================================================================
# 1. Sinusoidal Positional Timestep Embeddings
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Computes sinusoidal positional embeddings for the diffusion timestep t,
    similar to the Transformer architecture and Ho et al. (DDPM).
    Maps scalar timestep integers to a continuous high-dimensional vector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape (B,) containing integer or float timesteps
        Returns:
            Tensor of shape (B, dim) containing sinusoidal embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# ============================================================================
# 2. Residual Block with Timestep Conditioning
# ============================================================================

class ConvBlock(nn.Module):
    """
    Convolutional block with GroupNorm, SiLU activation, and
    timestep embedding projection injection.
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act1 = nn.SiLU()

        # Timestep projection: maps time_emb_dim -> out_channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Second conv branch
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()

        # Residual shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act1(self.norm1(self.conv1(x)))
        # Project time embedding and add to spatial feature map (B, C, 1, 1)
        time_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_proj
        h = self.act2(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


# ============================================================================
# 3. UNet Architecture for 28x28 Images
# ============================================================================

class DiffusionUNet(nn.Module):
    """
    U-Net architecture with residual blocks, skip connections, and
    sinusoidal timestep conditioning.
    Designed specifically for 28x28 single-channel (MNIST) image denoising.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, time_emb_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # Timestep MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        # Encoder (Downsampling)
        # Stage 1: 28x28 -> 14x14
        self.down1 = ConvBlock(32, 32, time_emb_dim)
        self.pool1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 28 -> 14

        # Stage 2: 14x14 -> 7x7
        self.down2 = ConvBlock(64, 64, time_emb_dim)
        self.pool2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 14 -> 7

        # Bottleneck: 7x7
        self.bot1 = ConvBlock(128, 128, time_emb_dim)
        self.bot2 = ConvBlock(128, 128, time_emb_dim)

        # Decoder (Upsampling)
        # Stage 2 up: 7x7 -> 14x14
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7 -> 14
        self.up_conv2 = ConvBlock(64 + 64, 64, time_emb_dim)

        # Stage 1 up: 14x14 -> 28x28
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 14 -> 28
        self.up_conv1 = ConvBlock(32 + 32, 32, time_emb_dim)

        # Final prediction layer: predicts epsilon (added noise)
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting noise epsilon.
        Args:
            x: (B, 1, 28, 28) noisy image tensor
            timestep: (B,) integer tensor of diffusion steps
        Returns:
            (B, 1, 28, 28) predicted noise tensor
        """
        # Time embedding
        t_emb = self.time_embed(timestep)

        # Initial feature extraction
        x0 = self.init_conv(x)  # (B, 32, 28, 28)

        # Down 1
        d1 = self.down1(x0, t_emb)      # (B, 32, 28, 28)
        p1 = self.pool1(d1)              # (B, 64, 14, 14)

        # Down 2
        d2 = self.down2(p1, t_emb)      # (B, 64, 14, 14)
        p2 = self.pool2(d2)              # (B, 128, 7, 7)

        # Bottleneck
        b1 = self.bot1(p2, t_emb)        # (B, 128, 7, 7)
        b2 = self.bot2(b1, t_emb)        # (B, 128, 7, 7)

        # Up 2 (skip connection from d2)
        u2 = self.up2(b2)                # (B, 64, 14, 14)
        u2 = torch.cat([u2, d2], dim=1)  # (B, 128, 14, 14)
        u2 = self.up_conv2(u2, t_emb)    # (B, 64, 14, 14)

        # Up 1 (skip connection from d1)
        u1 = self.up1(u2)                # (B, 32, 28, 28)
        u1 = torch.cat([u1, d1], dim=1)  # (B, 64, 28, 28)
        u1 = self.up_conv1(u1, t_emb)    # (B, 32, 28, 28)

        # Output predicted noise
        out = self.final_conv(u1)        # (B, 1, 28, 28)
        return out


# ============================================================================
# 4. Diffusion Process (Forward & Reverse Equations)
# ============================================================================

class GaussianDiffusion:
    """
    Implements the discrete-time Gaussian Diffusion process:
    - Forward diffusion: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1 - alpha_bar_t)*I)
    - Reverse diffusion: p_theta(x_{t-1} | x_t) using predicted noise epsilon_theta.
    """
    def __init__(self, timesteps: int = 300, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.timesteps = timesteps
        self.device = torch.device(device)

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for forward diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Below: log calculation clipped because the posterior variance is 0 at the beginning of the schedule
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(self.device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion: Samples x_t from q(x_t | x_0) in closed form.
        Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    def get_noise_parameters(self, t_int: int):
        """
        Returns the scalar values of alpha_bar and noise weighting for step t.
        Useful for displaying the exact mathematical values on the UI.
        """
        idx = max(0, min(t_int, self.timesteps - 1))
        alpha_bar = float(self.alphas_cumprod[idx].cpu().item())
        beta = float(self.betas[idx].cpu().item())
        signal_factor = math.sqrt(alpha_bar)
        noise_factor = math.sqrt(1.0 - alpha_bar)
        return {
            "timestep": idx,
            "total_timesteps": self.timesteps,
            "alpha_bar": round(alpha_bar, 4),
            "beta": round(beta, 6),
            "signal_weight": round(signal_factor, 4),
            "noise_weight": round(noise_factor, 4),
            "noise_percentage": round(noise_factor * 100, 1)
        }

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """
        Single step of reverse diffusion: samples x_{t-1} from p_theta(x_{t-1} | x_t).
        Formula:
            x_{t-1} = 1/sqrt(alpha_t) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred) + sigma_t * z
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise using the trained neural network
        predicted_noise = model(x_t, t_tensor)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        # Calculate mean mu_theta(x_t, t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            x_prev = mean + variance * noise
        else:
            x_prev = mean

        return x_prev

    @torch.no_grad()
    def reverse_diffusion(self, model: nn.Module, x_start_noisy: torch.Tensor, start_t: int, num_snapshots: int = 5):
        """
        Iterative reverse diffusion from timestep start_t down to 0.
        Collects intermediate snapshots along the denoising trajectory.

        Args:
            model: Trained PyTorch DiffusionUNet
            x_start_noisy: (1, 1, 28, 28) image at timestep start_t in [-1, 1]
            start_t: Starting timestep integer (0 to timesteps - 1)
            num_snapshots: Number of intermediate images to capture for UI visualization

        Returns:
            dict containing:
                - 'final_tensor': Clean estimated image tensor
                - 'snapshots': List of (step, tensor) tuples
        """
        model.eval()
        x_curr = x_start_noisy.clone().to(self.device)

        # Determine snapshot capture points
        snapshot_steps = set()
        if num_snapshots > 0 and start_t > 0:
            snapshot_steps = set(int(round(s)) for s in np.linspace(start_t, 0, num_snapshots))

        snapshots = []
        if start_t in snapshot_steps:
            snapshots.append((start_t, x_curr.clone()))

        # Iteratively denoise from start_t down to 0
        for step in reversed(range(start_t + 1)):
            x_curr = self.p_sample(model, x_curr, step)
            # Clamp to valid image dynamic range
            x_curr = torch.clamp(x_curr, -1.0, 1.0)

            if step in snapshot_steps and step != start_t:
                snapshots.append((step, x_curr.clone()))

        if 0 not in [s[0] for s in snapshots]:
            snapshots.append((0, x_curr.clone()))

        return {
            "final_tensor": x_curr,
            "snapshots": snapshots
        }


# ============================================================================
# 5. Image Transformation & Evaluation Helpers
# ============================================================================

def pil_to_tensor(image: Image.Image, device: str = "cpu") -> torch.Tensor:
    """
    Converts a PIL image to a normalized (1, 1, 28, 28) PyTorch tensor in range [-1, 1].
    Converts to grayscale and resizes using high quality Lanczos resampling.
    """
    # Convert to grayscale
    gray_img = image.convert("L")
    # Resize to MNIST dimension 28x28
    resized_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(resized_img, dtype=np.float32)

    # Normalize from [0, 255] to [-1.0, 1.0]
    arr = (arr / 127.5) - 1.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Converts a normalized (1, 1, 28, 28) or (28, 28) PyTorch tensor in range [-1, 1]
    back to a PIL Image in range [0, 255].
    """
    t = tensor.detach().cpu().squeeze()
    # Denormalize from [-1.0, 1.0] to [0.0, 255.0]
    t = (t + 1.0) * 127.5
    t = torch.clamp(t, 0.0, 255.0)
    arr = t.numpy().astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def calculate_metrics(clean_tensor: torch.Tensor, denoised_tensor: torch.Tensor):
    """
    Calculates quantitative image quality metrics:
    - MSE (Mean Squared Error)
    - PSNR (Peak Signal-to-Noise Ratio in dB)
    - Noise Reduction percentage
    """
    c = clean_tensor.detach().cpu().squeeze().numpy()
    d = denoised_tensor.detach().cpu().squeeze().numpy()

    # Map from [-1, 1] to [0, 1] for metric calculation
    c_norm = (c + 1.0) / 2.0
    d_norm = (d + 1.0) / 2.0

    mse = float(np.mean((c_norm - d_norm) ** 2))
    if mse < 1e-10:
        psnr = 100.0
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))

    # Signal correlation
    c_std = np.std(c_norm)
    d_std = np.std(d_norm)
    if c_std > 1e-5 and d_std > 1e-5:
        correlation = float(np.mean((c_norm - np.mean(c_norm)) * (d_norm - np.mean(d_norm))) / (c_std * d_std))
    else:
        correlation = 1.0

    return {
        "mse": round(mse, 5),
        "psnr_db": round(psnr, 2),
        "correlation": round(max(0.0, min(1.0, correlation)), 4)
    }
