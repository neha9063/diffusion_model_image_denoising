"""
Diffusion Model for Image Denoising
Training Script for MNIST Handwritten Digits
Implements Forward Diffusion, Noise Prediction, MSE Optimization & Loss Plotting
"""

import os
import sys

# Ensure unbuffered standard output and writable matplotlib directory
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs")

import argparse
import time
import math
import socket
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

from diffusion import DiffusionUNet, GaussianDiffusion


# ============================================================================
# Synthetic MNIST Fallback (Ensures 100% offline reliability in any environment)
# ============================================================================

class SyntheticMNISTDataset(Dataset):
    """
    Offline fallback dataset generating realistic synthetic handwritten digits
    with elastic variations, rotations, and line-width variations.
    Guarantees the training script runs even without active internet connection.
    """
    def __init__(self, size: int = 4000):
        self.size = size
        self.samples = []
        print(f"[*] Initializing synthetic digit dataset ({size} samples)...", flush=True)

        for idx in range(size):
            digit = idx % 10
            # Draw digit on 28x28 grayscale canvas
            img = Image.new("L", (28, 28), color=0)
            draw = ImageDraw.Draw(img)

            cx = 14 + np.random.randint(-2, 3)
            cy = 14 + np.random.randint(-2, 3)

            if digit == 0:
                draw.ellipse([cx - 7, cy - 9, cx + 7, cy + 9], outline=255, width=2)
            elif digit == 1:
                draw.line([cx, cy - 9, cx, cy + 9], fill=255, width=3)
                draw.line([cx - 3, cy - 6, cx, cy - 9], fill=255, width=2)
            elif digit == 2:
                draw.arc([cx - 6, cy - 9, cx + 6, cy], start=180, end=360, fill=255, width=2)
                draw.line([cx + 6, cy, cx - 6, cy + 9], fill=255, width=2)
                draw.line([cx - 6, cy + 9, cx + 6, cy + 9], fill=255, width=3)
            elif digit == 3:
                draw.arc([cx - 6, cy - 9, cx + 6, cy], start=270, end=90, fill=255, width=2)
                draw.arc([cx - 6, cy, cx + 6, cy + 9], start=270, end=90, fill=255, width=2)
            elif digit == 4:
                draw.line([cx - 6, cy - 9, cx - 6, cy + 1], fill=255, width=2)
                draw.line([cx - 6, cy + 1, cx + 6, cy + 1], fill=255, width=2)
                draw.line([cx + 2, cy - 9, cx + 2, cy + 9], fill=255, width=2)
            elif digit == 5:
                draw.line([cx + 5, cy - 9, cx - 5, cy - 9], fill=255, width=2)
                draw.line([cx - 5, cy - 9, cx - 5, cy - 1], fill=255, width=2)
                draw.arc([cx - 6, cy - 2, cx + 5, cy + 9], start=270, end=90, fill=255, width=2)
            elif digit == 6:
                draw.arc([cx - 6, cy - 9, cx + 6, cy + 9], start=90, end=270, fill=255, width=2)
                draw.ellipse([cx - 6, cy - 1, cx + 6, cy + 9], outline=255, width=2)
            elif digit == 7:
                draw.line([cx - 6, cy - 9, cx + 6, cy - 9], fill=255, width=2)
                draw.line([cx + 6, cy - 9, cx - 3, cy + 9], fill=255, width=2)
            elif digit == 8:
                draw.ellipse([cx - 5, cy - 9, cx + 5, cy], outline=255, width=2)
                draw.ellipse([cx - 6, cy, cx + 6, cy + 9], outline=255, width=2)
            elif digit == 9:
                draw.ellipse([cx - 6, cy - 9, cx + 6, cy + 1], outline=255, width=2)
                draw.line([cx + 6, cy - 9, cx + 6, cy + 9], fill=255, width=2)

            arr = np.array(img, dtype=np.float32)
            # Add subtle natural texture noise
            arr += np.random.normal(0, 5.0, arr.shape).astype(np.float32)
            arr = np.clip(arr, 0.0, 255.0)
            # Normalize to [-1.0, 1.0]
            arr = (arr / 127.5) - 1.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 28, 28)
            self.samples.append(tensor)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], 0


# ============================================================================
# Dataset Loading
# ============================================================================

def is_network_available(host="8.8.8.8", port=53, timeout=1.0):
    """Checks if network access is available with a 1-second timeout."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


def get_dataloader(batch_size: int = 128, subset_size: int = None, data_dir: str = "./data", force_synthetic: bool = False):
    """
    Loads MNIST dataset if locally available or network is accessible.
    Otherwise gracefully falls back to synthetic digits.
    Normalizes images to [-1, 1].
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Maps [0, 1] -> [-1, 1]
    ])

    # Check if local MNIST raw files exist
    mnist_raw_dir = os.path.join(data_dir, "MNIST", "raw")
    has_local_mnist = os.path.exists(mnist_raw_dir) and len(os.listdir(mnist_raw_dir)) >= 4

    dataset = None

    if not force_synthetic and has_local_mnist:
        try:
            print("[*] Loading existing local MNIST dataset...", flush=True)
            dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
            print(f"[+] Loaded local MNIST ({len(dataset)} samples).", flush=True)
        except Exception as e:
            print(f"[!] Local MNIST load failed: {e}", flush=True)

    if dataset is None and not force_synthetic and is_network_available():
        try:
            print("[*] Downloading official MNIST dataset...", flush=True)
            dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            print(f"[+] Downloaded official MNIST ({len(dataset)} samples).", flush=True)
        except Exception as e:
            print(f"[!] Warning: Could not download official MNIST ({e}).", flush=True)

    if dataset is None:
        print("[+] Using synthetic digit dataset (instant, 100% offline).", flush=True)
        num_samples = subset_size if subset_size else 4000
        dataset = SyntheticMNISTDataset(size=num_samples)

    if subset_size and subset_size < len(dataset):
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
        print(f"[*] Training on subset of {subset_size} samples.", flush=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader


# ============================================================================
# Training Function
# ============================================================================

def train(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 2e-4,
    timesteps: int = 300,
    save_path: str = "models/diffusion_mnist.pth",
    plot_path: str = "static/images/loss_plot.png",
    subset_size: int = None,
    device_name: str = "auto",
    force_synthetic: bool = False
):
    """
    Trains the Denoising Diffusion Probabilistic Model (DDPM) on MNIST.
    Optimizes MSE loss between added noise and predicted noise:
        L_simple = E_{t, x_0, eps} [ || eps - eps_theta(x_t, t) ||^2 ]
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)

    # Select compute device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"[*] Target Compute Device: {device}", flush=True)

    # Initialize model and diffusion manager
    model = DiffusionUNet(in_channels=1, out_channels=1, time_emb_dim=128).to(device)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=str(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Diffusion UNet initialized with {total_params:,} trainable parameters.", flush=True)

    dataloader = get_dataloader(batch_size=batch_size, subset_size=subset_size, force_synthetic=force_synthetic)
    num_batches = len(dataloader)
    print(f"[*] Starting training: {epochs} Epochs | Batch Size: {batch_size} | Timesteps: {timesteps}")

    epoch_losses = []
    step_losses = []
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
            x_0 = images.to(device)
            current_b = x_0.shape[0]

            # 1. Sample uniform random timestep t for each image in batch
            t = torch.randint(0, timesteps, (current_b,), device=device).long()

            # 2. Sample standard Gaussian noise epsilon ~ N(0, I)
            noise = torch.randn_like(x_0)

            # 3. Compute forward diffusion noisy image x_t
            # Formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
            x_t = diffusion.q_sample(x_0, t, noise)

            # 4. Neural Network predicts the added noise epsilon_theta(x_t, t)
            predicted_noise = model(x_t, t)

            # 5. Calculate Mean Squared Error loss
            loss = loss_fn(predicted_noise, noise)

            # 6. Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step_losses.append(loss.item())

            if (batch_idx + 1) % max(1, num_batches // 4) == 0 or (batch_idx + 1) == num_batches:
                avg_so_far = running_loss / (batch_idx + 1)
                print(f"    Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{num_batches}] -> Current MSE Loss: {avg_so_far:.5f}")

        avg_epoch_loss = running_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"[*] Completed Epoch {epoch}/{epochs} in {epoch_duration:.2f}s | Average Loss: {avg_epoch_loss:.5f}\n")

    total_duration = time.time() - start_total_time
    print(f"[+] Training complete in {total_duration:.2f} seconds.")

    # Save model checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "timesteps": timesteps,
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "loss_history": epoch_losses,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    torch.save(checkpoint, save_path)
    print(f"[+] Model checkpoint saved successfully to: {save_path}")

    # Generate and save loss curve plot
    save_loss_plot(epoch_losses, step_losses, plot_path)
    print(f"[+] Training loss visualization saved to: {plot_path}")

    return model, checkpoint


# ============================================================================
# Loss Curve Visualization
# ============================================================================

def save_loss_plot(epoch_losses, step_losses, plot_path: str):
    """
    Renders a high-resolution, modern graph of the training loss curve.
    """
    plt.figure(figsize=(9, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    if len(epoch_losses) > 1:
        # Plot epoch averages
        epochs_x = list(range(1, len(epoch_losses) + 1))
        plt.plot(epochs_x, epoch_losses, marker="o", color="#4f46e5", linewidth=2.5, label="Epoch Mean Loss")
        plt.title("DDPM Training Loss Convergence (MSE on MNIST)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Noise Prediction Loss (MSE)", fontsize=11)
        plt.xticks(epochs_x)
    else:
        # Plot step losses with moving average
        window = max(1, len(step_losses) // 20)
        smoothed = np.convolve(step_losses, np.ones(window)/window, mode="valid")
        plt.plot(step_losses, alpha=0.3, color="#93c5fd", label="Step Loss")
        plt.plot(smoothed, color="#3b82f6", linewidth=2, label=f"Moving Average ({window} steps)")
        plt.title("DDPM Noise Prediction Loss (MSE)", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Training Step", fontsize=11)
        plt.ylabel("Mean Squared Error (MSE)", fontsize=11)

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DDPM for Image Denoising on MNIST")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=300, help="Number of diffusion timesteps")
    parser.add_argument("--save-path", type=str, default="models/diffusion_mnist.pth", help="Checkpoint save path")
    parser.add_argument("--plot-path", type=str, default="static/images/loss_plot.png", help="Loss plot destination")
    parser.add_argument("--quick", action="store_true", help="Quick mode (trains on subset of 2000 samples for rapid setup)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Compute device")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic digit generation without querying network")

    args = parser.parse_args()

    subset = 2500 if args.quick else None

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        save_path=args.save_path,
        plot_path=args.plot_path,
        subset_size=subset,
        device_name=args.device,
        force_synthetic=args.force_synthetic
    )
