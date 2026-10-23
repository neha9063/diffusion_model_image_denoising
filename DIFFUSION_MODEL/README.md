# Diffusion Model for Image Denoising

**B.Tech Artificial Intelligence & Machine Learning Mini Project**

A deep learning web application demonstrating **Forward Diffusion** (Gaussian noise injection via a linear variance schedule) and **Reverse Diffusion** (iterative noise estimation and removal) using a trained **PyTorch U-Net** on handwritten digit images (MNIST).

---

## 📌 Project Overview

Diffusion models (such as DDPM — *Denoising Diffusion Probabilistic Models*) represent state-of-the-art generative modeling. Unlike traditional image filters (Gaussian blur, bilateral filter, median filter) that indiscriminately blur high-frequency details, diffusion models learn the underlying probability distribution of the data manifold $\nabla \log p(x)$ and iteratively reconstruct clean images from noise.

### Key Highlights
- **Genuine Deep Learning**: Noise prediction using a custom 1.18M parameter PyTorch U-Net with Sinusoidal Positional Timestep Embeddings and residual blocks with GroupNorm.
- **Side-by-Side Comparison**: Real-time visualization of clean ($x_0$), noisy ($x_t$), and denoised ($\hat{x}_0$) images.
- **Progressive Trajectory Filmstrip**: Inspect intermediate reconstructed states as reverse diffusion steps down from $t \to 0$.
- **Instant Testing Gallery**: 1-click test gallery containing pre-generated MNIST digits (0–9) so anyone can test without uploading.
- **Quantitative Metrics**: Instant calculation of Peak Signal-to-Noise Ratio (PSNR in dB), Mean Squared Error (MSE), structural correlation, and inference latency in milliseconds.
- **100% Offline & CPU Compatible**: Runs locally on standard laptops without requiring a GPU or active internet connection.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, FontAwesome |
| **Backend** | Python 3, Flask 3.0 |
| **Deep Learning Engine** | PyTorch 2.x, Torchvision |
| **Image Processing & Math** | NumPy, Pillow (PIL), SciPy |
| **Data Visualization** | Matplotlib (Agg headless backend) |
| **Dataset** | MNIST Handwritten Digits (28×28 grayscale, normalized to $[-1, 1]$) |

---

## 📁 Project Directory Structure

```
diffusion-image-denoising/
│
├── app.py                     # Flask web server and REST API endpoints
├── diffusion.py               # U-Net architecture & GaussianDiffusion math implementation
├── train.py                   # PyTorch training script with offline synthetic fallback
├── generate_samples.py        # Utility to generate sample digit images (0-9)
├── test_app.py                # Automated integration test suite
├── requirements.txt           # Python package dependencies
├── README.md                  # Complete project setup and technical guide
├── PROJECT_REPORT.md          # Comprehensive B.Tech academic mini-project report
├── VIVA_VOCE_DEMO_SCRIPT.md   # Demonstration script & Viva Voce Q&A guide
│
├── models/
│   └── diffusion_mnist.pth    # Trained PyTorch DDPM weights checkpoint
│
├── uploads/                   # Temporary storage for uploaded and normalized images
├── outputs/                   # Generated noisy images, denoised outputs, and temp plots
│
├── templates/
│   ├── index.html             # Interactive Denoising Studio home page
│   ├── learn.html             # Conceptual educational guide & math breakdown
│   └── results.html           # Evaluation dashboard, metrics table & loss curve
│
└── static/
    ├── css/
    │   └── style.css          # AI-themed modern responsive stylesheet
    ├── js/
    │   └── script.js          # Interactive frontend controller & AJAX logic
    └── images/
        ├── loss_plot.png      # Matplotlib loss convergence graph
        └── samples/           # Pre-loaded sample digits (digit_0.png to digit_9.png)
```

---

## 🔬 Mathematical Formulation

### 1. Forward Diffusion Process ($q$)
Adds Gaussian noise to clean image $x_0$ according to a variance schedule $\beta_1, \dots, \beta_T$:
$$\beta_t = \beta_1 + \frac{t - 1}{T - 1} (\beta_T - \beta_1)$$
$$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$

Closed-form sampling at any arbitrary timestep $t$:
$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

### 2. Neural Network Noise Prediction ($\epsilon_\theta$)
A U-Net parameterized by weights $\theta$ is conditioned on spatial image $x_t$ and scalar timestep $t$ mapped through sinusoidal embeddings:
$$\text{Loss} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 3. Reverse Diffusion Process ($p_\theta$)
Iteratively steps backwards from $x_t$ to $x_0$:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
where $\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}$ for $t > 1$, and $z = 0$ at $t = 1$.

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Open Terminal in Project Directory
Open PowerShell, Command Prompt, or terminal and navigate to the project directory:
```bash
cd path/to/DIFFUSION_MODEL
```

### Step 2: Install Python Dependencies
Install required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (Optional — Checkpoint Already Included)
A pre-trained checkpoint `models/diffusion_mnist.pth` and loss curve `static/images/loss_plot.png` are already included!
To re-train or train with custom parameters:
```bash
# Rapid training (runs in ~1 minute on CPU):
python train.py --epochs 3 --quick

# Full training (standard MNIST dataset):
python train.py --epochs 5 --batch-size 128 --lr 0.0002
```

### Step 4: Run the Automated Test Suite
Verify that all components (PyTorch model, endpoints, math functions) are functioning properly:
```bash
python test_app.py
```
*(All 6 unit and integration tests should pass with `OK`.)*

### Step 5: Start the Flask Web Server
Run the Flask server:
```bash
python app.py
```
You will see output indicating the server is running:
```
[*] Loading model checkpoint from models/diffusion_mnist.pth...
[+] Model loaded successfully on cpu!
 * Running on http://127.0.0.1:5000
```

### Step 6: Open in Browser
Open your web browser (Chrome, Edge, Firefox, Brave) and visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🖥️ How to Use the Application

1. **Select or Upload Image**:
   - Click any digit in the **Instant Test Gallery (0–9)** at the top left.
   - Or drag-and-drop / upload any image from your computer. The app automatically converts it to 28×28 grayscale and normalizes pixel values to $[-1, 1]$.
2. **Configure Forward Noise**:
   - Adjust the **Diffusion Timestep Slider ($t$)** from $1$ to $300$, or click one of the preset buttons (*Light*, *Moderate*, *Heavy*, *Extreme*).
   - Observe how the mathematical values for $\sqrt{\bar{\alpha}_t}$ (signal weight) and $\sqrt{1 - \bar{\alpha}_t}$ (noise factor) update dynamically.
   - Click **"Inject Gaussian Noise"** to view the noisy image $x_t$.
3. **Execute AI Reverse Denoising**:
   - Click **"Run Reverse Diffusion"**.
   - Watch the animated progress indicator as the PyTorch U-Net samples backwards through the Markov chain.
   - View the recovered image $\hat{x}_0$ side-by-side with the original and noisy versions.
   - Inspect the **Step-by-Step Trajectory Filmstrip** showing the progressive reconstruction frames.
   - Check quantitative metrics: **Inference Time (ms)**, **MSE Loss**, and **PSNR (dB)**.
   - Click **"Download"** to save the reconstructed output image.
4. **Explore the Learn & Results Pages**:
   - Click **"Learn Diffusion"** in the navigation bar to read the conceptual guide, compare traditional filters vs. diffusion, and study the mathematical formulation.
   - Click **"Results & Loss"** to view the training loss curve and the session log table.

---

## ❓ Troubleshooting & FAQs

- **Issue: `ModuleNotFoundError: No module named 'torch'`**
  - *Fix*: Run `pip install torch torchvision` or check that your active Python environment matches the one where packages were installed.
- **Issue: Model checkpoint missing warning on web page**
  - *Fix*: Run `python train.py --epochs 3 --quick` in your terminal to train and save `models/diffusion_mnist.pth`.
- **Issue: Uploaded photo looks distorted**
  - *Explanation*: The model is trained specifically on handwritten 28×28 grayscale digits. Uploaded images are resized and projected onto this digit manifold. For high-resolution natural photographs, multi-channel models (trained on datasets like CelebA or FFHQ) are required.

---

## 📜 License & Academic Integrity
Developed as a B.Tech Artificial Intelligence & Machine Learning Mini Project. Free to use and adapt for educational and demonstration purposes.
