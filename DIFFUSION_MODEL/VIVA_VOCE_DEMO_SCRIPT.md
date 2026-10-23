# Mini-Project Demonstration Script & Viva Voce Q&A Guide

**Project Title:** Diffusion Model for Image Denoising  
**Candidate Target:** B.Tech AI & ML External & Internal Lab Viva / Project Presentation  

---

## Part 1: Step-by-Step Live Demonstration Script (5-Minute Walkthrough)

When presenting to professors or examiners, follow this structured script while navigating the web application:

### Step 1: Introduction (0:00 – 1:00)
- **Action:** Open `http://127.0.0.1:5000` in the browser. Point to the top navigation bar and hero section.
- **What to say:**
  > *"Good morning respected examiners. Our mini project is titled **Diffusion Model for Image Denoising**. The objective is to demonstrate how modern deep generative models—specifically Denoising Diffusion Probabilistic Models (DDPM)—destroy image structure through forward diffusion and learn to reconstruct crisp, clean images through reverse diffusion using a trained PyTorch U-Net neural network."*
- **Highlight:** Point out the live badge in the navbar: `PyTorch UNet Ready (300 steps)`. Explain that the neural network weights are actively loaded into memory on CPU.

---

### Step 2: Input Selection & Image Upload (1:00 – 1:45)
- **Action:** Click one of the digit tiles in the **Instant Test Gallery** (e.g., `Digit 3` or `Digit 7`), or drag-and-drop a custom image into the upload box.
- **What to say:**
  > *"Here in Step 1, the user can upload a custom PNG/JPG image or choose from our pre-loaded digit gallery. The backend automatically converts the image to grayscale, resizes it to the standard $28 \times 28$ input resolution, and normalizes pixel values to the dynamic range $[-1.0, +1.0]$, which matches the mathematical formulation of our diffusion model."*

---

### Step 3: Forward Diffusion Demonstration (1:45 – 2:45)
- **Action:** Slide the **Timestep Slider ($t$)** to `150`, then point to the **Mathematical Parameters Box**. Click **"Inject Gaussian Noise"**.
- **What to say:**
  > *"In Step 2, we simulate forward diffusion. In traditional Markov chains, noise is added sequentially one step at a time. However, using the reparameterization trick, we can jump to any arbitrary timestep $t$ in a single closed-form calculation:*
  > $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
  > *As we slide the timestep slider to $t = 150$, you can see the signal retention factor $\sqrt{\bar{\alpha}_t}$ decrease while the noise factor $\sqrt{1 - \bar{\alpha}_t}$ increases. Clicking 'Inject Gaussian Noise' adds real Gaussian noise and displays the resulting noisy image $x_t$."*

---

### Step 4: Reverse Diffusion & Reconstruction (2:45 – 4:00)
- **Action:** Click **"Run Reverse Diffusion"**. Watch the animated progress status update.
- **What to say:**
  > *"Now we execute reverse diffusion in Step 3. Our system does NOT use a simple blur or median filter. Instead, our trained 1.18-million-parameter PyTorch U-Net model takes the noisy image $x_t$ and scalar timestep $t$, predicts the added noise $\epsilon_\theta(x_t, t)$, and iteratively subtracts it across all timesteps down to $t = 0$.*
  > *Notice the **Reverse Diffusion Step-by-Step Trajectory Filmstrip** below: it captures snapshots along the Markov chain, showing how stochastic noise gradually condenses into a sharp, legible digit."*
- **Highlight:** Point out the quantitative metrics:
  - **Inference Time:** $\approx 700 \text{ ms}$.
  - **MSE Loss:** Low Mean Squared Error between $x_0$ and $\hat{x}_0$.
  - **PSNR:** Peak Signal-to-Noise Ratio (typically $>17\text{--}25 \text{ dB}$).
  - Click the **"Download"** button to show that the output is exportable.

---

### Step 5: Educational & Results Dashboards (4:00 – 5:00)
- **Action:** Click **"Learn Diffusion"** in the navbar, show the comparison table, then click **"Results & Loss"** to display the training loss curve.
- **What to say:**
  > *"To document the project academically, we built a **Learn Diffusion** page that compares traditional filters (Gaussian blur, bilateral, Wiener) against generative diffusion models, explaining why traditional filters fail at high noise levels. Finally, the **Results Dashboard** shows our training loss convergence curve, where the noise prediction MSE steadily dropped from 1.06 down to 0.18 over epochs."*

---

## Part 2: Top 10 Viva Voce Questions & High-Scoring Answers

### Q1: What is the core difference between Forward Diffusion and Reverse Diffusion?
**Answer:**
> **Forward Diffusion ($q$)** is a fixed, non-trainable Markov process that gradually corrupts a clean image by adding Gaussian noise over $T$ discrete timesteps until it becomes isotropic noise.  
> **Reverse Diffusion ($p_\theta$)** is the learned generative process. A deep neural network (U-Net) is trained to estimate the noise added at each timestep, enabling us to step backward from pure noise to recover the clean original image.

---

### Q2: Why does the network predict the noise $\epsilon$ instead of predicting the clean image $x_0$ directly?
**Answer:**
> As proven by Ho et al. (2020), predicting the noise vector $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ parameterizes the model as a **score-based estimator** ($\nabla_x \log p_t(x)$). Noise has zero mean and unit variance, which is a stationary target across all timesteps $t$. Predicting $x_0$ directly leads to severe oversmoothing because the network averages multiple plausible high-frequency modes.

---

### Q3: What is the closed-form equation for forward diffusion, and why is it important?
**Answer:**
> The equation is:
> $$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$
> where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.  
> **Significance:** Without this closed form, training would require sequentially computing $t$ Markov transitions step by step. With this closed form, we can directly sample $x_t$ at any random timestep $t$ in a single operation, accelerating training by orders of magnitude.

---

### Q4: Why is a U-Net architecture preferred over a standard CNN?
**Answer:**
> Diffusion models require predicting noise of the **exact same spatial dimension** as the input ($28 \times 28 \times 1$). A U-Net provides:
> 1. An **encoder-decoder structure** to capture high-level semantic context at lower resolutions ($7 \times 7$).
> 2. **Skip connections** that concatenate low-level feature maps from the encoder directly into the decoder. This preserves fine spatial details and pixel-level gradients necessary for high-fidelity noise subtraction.

---

### Q5: How is the timestep $t$ conditioned into the neural network?
**Answer:**
> The scalar integer $t$ is mapped into a 128-dimensional continuous vector using **Sinusoidal Positional Embeddings** (derived from the Transformer architecture):
> $$PE(t, 2i) = \sin(t / 10000^{2i/d}), \quad PE(t, 2i+1) = \cos(t / 10000^{2i/d})$$
> This embedding is processed through a 2-layer MLP and linearly projected to match the channel dimension of each residual convolutional block, where it is added directly to the spatial feature maps.

---

### Q6: Why did you use Group Normalization instead of Batch Normalization?
**Answer:**
> Batch Normalization computes statistics across the mini-batch dimension. During web inference, single images are processed with a batch size of $1$, which causes Batch Normalization statistics to degrade or produce numerical instability. **Group Normalization (GN)** divides channels into groups (e.g., 8 groups) and computes statistics independently per sample, guaranteeing identical performance during both training (batch size 128) and real-time inference (batch size 1).

---

### Q7: Why can't we just use a Gaussian Blur or Median Filter for denoising?
**Answer:**
> Traditional spatial filters are **local smoothing operators**:
> - **Gaussian Blur** convolves neighboring pixels, which acts as a low-pass filter, permanently destroying sharp edges and stroke contours.
> - **Median / Bilateral Filters** preserve edges under low noise but cannot "hallucinate" or generate missing structure when the noise level is high ($t > 150$).
> - **Diffusion Models** are generative: they have learned the prior distribution of digits and can reconstruct intact strokes even when the input is heavily corrupted.

---

### Q8: What loss function is used, and why is MSE appropriate?
**Answer:**
> We use the simplified Mean Squared Error loss:
> $$L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$
> Because the ground-truth added noise $\epsilon$ is distributed as standard Gaussian $\mathcal{N}(0, \mathbf{I})$, optimizing MSE corresponds to maximizing the variational lower bound (ELBO) on data log-likelihood while maintaining training stability.

---

### Q9: What is the main trade-off of Diffusion Models compared to GANs?
**Answer:**
> - **Advantage over GANs:** Diffusion models do not suffer from mode collapse, do not require adversarial min-max balancing, and provide stable training dynamics with full coverage of the data distribution.
> - **Disadvantage (Trade-off):** Inference speed. GANs generate an image in a single forward pass, whereas diffusion models require an iterative loop of $T$ sequential network evaluations (e.g., 100–300 steps).

---

### Q10: Why is this project trained on MNIST, and what happens if a colorful real-world photo is uploaded?
**Answer:**
> **Why MNIST:** MNIST (28×28 grayscale digits) allows training a high-quality 1.18M-parameter model on a standard CPU in minutes, enabling instant sub-second interactive reverse diffusion during viva demonstrations.  
> **Real-world photos:** If a user uploads a high-resolution natural photo, our app converts and resizes it to 28×28 grayscale and projects it onto the digit manifold. The model will denoise it according to its learned digit stroke priors. Denoising arbitrary natural images requires training larger multi-channel models on diverse natural datasets such as CelebA, CIFAR-10, or ImageNet.
