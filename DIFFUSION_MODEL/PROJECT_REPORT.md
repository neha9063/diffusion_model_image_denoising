# B.Tech AI & ML Project Report Outline

## Title: Diffusion Model for Image Denoising

**Academic Degree:** Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning  
**Domain:** Deep Generative Modeling & Computer Vision  

---

## Abstract
Image denoising is a fundamental challenge in computer vision and digital image processing. Conventional spatial and frequency-domain filtering techniques (such as Gaussian filtering, Wiener filters, and bilateral filtering) suffer from structural oversmoothing and an inability to recover lost high-frequency details. Recent advancements in deep generative modeling have established **Denoising Diffusion Probabilistic Models (DDPM)** as the state of the art, surpassing Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) in fidelity and mode coverage.

In this project, a complete end-to-end web-based deep learning system is developed to demonstrate the mechanics of image diffusion. The system implements:
1. A **Forward Diffusion Process** that systematically destroys spatial coherence through a parameterized Markov chain using Gaussian noise injection governed by a linear variance schedule.
2. A **Reverse Diffusion Process** powered by a 1.18-million-parameter deep **U-Net architecture** conditioned on sinusoidal timestep embeddings, trained on the MNIST dataset to predict added noise $\epsilon_\theta(x_t, t)$.
3. An interactive, responsive **Web Application** built using Flask and Bootstrap 5, featuring real-time mathematical parameter inspection, step-by-step reverse trajectory visualization, quantitative reconstruction metrics (MSE, PSNR, structural correlation), and a comparative study against conventional filters.

---

## Table of Contents
1. **Chapter 1: Introduction**
   - 1.1 Background & Motivation
   - 1.2 Problem Statement
   - 1.3 Project Objectives
   - 1.4 Scope of the Project
2. **Chapter 2: Literature Survey**
   - 2.1 Classical Image Denoising Methods
   - 2.2 Deep Learning Approaches (Autoencoders & GANs)
   - 2.3 Evolution of Diffusion Probabilistic Models
3. **Chapter 3: Theoretical & Mathematical Modeling**
   - 3.1 The Forward Diffusion Process (Markov Chain)
   - 3.2 Closed-Form Sampling ($q(x_t | x_0)$)
   - 3.3 The Reverse Process & Score Matching
   - 3.4 Simplified Training Objective ($L_{\text{simple}}$)
4. **Chapter 4: System Architecture & Implementation**
   - 4.1 Overall System Architecture
   - 4.2 U-Net Neural Network Specification
   - 4.3 Sinusoidal Positional Timestep Embeddings
   - 4.4 Flask Backend & RESTful Endpoints
   - 4.5 Interactive User Interface
5. **Chapter 5: Experimental Results & Performance Analysis**
   - 5.1 Training Convergence & Loss Curve
   - 5.2 Qualitative Denoising Results
   - 5.3 Quantitative Performance Metrics (PSNR & MSE)
   - 5.4 Sampling Latency Analysis
6. **Chapter 6: Comparative Analysis & Discussion**
   - 6.1 Traditional Filters vs. Diffusion Models
   - 6.2 Model Limitations on Natural Photographs
7. **Chapter 7: Conclusion & Future Scope**
   - 7.1 Summary of Contributions
   - 7.2 Future Enhancements (Latent Diffusion & CFG)
8. **References**

---

## Detailed Chapter Outlines

### Chapter 1: Introduction
- **1.1 Background:** Digital images acquired through optical sensors or transmitted over communication channels often suffer from additive Gaussian noise. While classical methods treat denoising as an inverse filtering problem, modern generative models view it as stochastic sampling along the data distribution score.
- **1.2 Problem Statement:** Design, train, and deploy an accessible, interactive deep learning platform that demonstrates forward noise injection and reverse generative denoising on image data using PyTorch and Flask.
- **1.3 Objectives:**
  - Formulate mathematically consistent discrete-time Gaussian diffusion equations.
  - Implement a timestep-conditioned U-Net architecture.
  - Train the network on MNIST digits using mean squared error noise prediction loss.
  - Create a web dashboard allowing users to manipulate noise timesteps $t$ and visualize step-by-step reconstruction.

---

### Chapter 2: Literature Survey
- **Classical Filters:** Gaussian blurring applies spatial convolution with a bell curve, eliminating noise at the cost of blurring object edges. Bilateral filters preserve edges but cannot generate missing pixel content when noise variance is high.
- **Autoencoders:** Denoising autoencoders map noisy images directly to clean outputs, but frequently produce blurry, average reconstructions due to the $L_2$ regression loss over multimodal distributions.
- **Generative Adversarial Networks (GANs):** GANs generate sharp images but suffer from training instability, mode collapse, and hyperparameter sensitivity.
- **Diffusion Models (Sohl-Dickstein et al., 2015; Ho et al., 2020):** DDPM establishes a stable, likelihood-based framework with training objectives that optimize directly over Gaussian noise prediction without adversarial dynamics.

---

### Chapter 3: Mathematical Modeling
- **Linear Variance Schedule:**
  $$\beta_t = \beta_1 + \frac{t - 1}{T - 1}(\beta_T - \beta_1), \quad \beta_1 = 10^{-4}, \, \beta_T = 0.02, \, T = 300$$
- **Signal & Noise Coefficients:**
  $$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^t \alpha_s$$
- **Closed-Form Forward Jump:**
  $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$
- **Reverse Iteration:**
  $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$
  where $\sigma_t = \sqrt{\tilde{\beta}_t} = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t}$.

---

### Chapter 4: System Architecture & Implementation
- **Neural Network Details:**
  - Architecture: Symmetric U-Net with skip connections.
  - Downsampling: $28\times 28 \to 14\times 14 \to 7\times 7$ via strided convolutions.
  - Bottleneck: Two convolutional blocks with 128 feature channels.
  - Upsampling: Transposed convolutions ($7\times 7 \to 14\times 14 \to 28\times 28$).
  - Normalization: Group Normalization (`num_groups=8`) ensuring stable inference for batch size 1.
  - Activation: SiLU (Swish) non-linearities for smooth gradient flow.
- **Timestep Embedding:**
  $$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$
  Followed by a 2-layer MLP projected into each convolutional layer.

---

### Chapter 5: Experimental Results
- **Loss Convergence:**
  - Epoch 1 Average Loss: 0.58299
  - Epoch 2 Average Loss: 0.24030
  - Epoch 3 Average Loss: 0.18740 (Steady monotonic convergence).
- **Quantitative Metrics:**
  - Peak Signal-to-Noise Ratio (PSNR): Typically ranges between $16.5 \text{ dB}$ to $28.0 \text{ dB}$ depending on timestep $t$.
  - Mean Squared Error (MSE): $< 0.02$ on moderate noise levels.
  - Inference Latency: $\approx 600\text{--}900 \text{ ms}$ on CPU for $t=60$ steps, making it practical for real-time interactive demonstrations.

---

### Chapter 6: Discussion & Model Limitations
- **Resolution & Manifold Constraint:** The network is optimized specifically for $28\times 28$ single-channel handwritten digits. High-resolution color photos projected onto this manifold undergo spatial downsampling and grayscale conversion.
- **Inference Speed:** Unlike single-pass feedforward autoencoders, reverse diffusion requires $t$ sequential evaluation steps. Accelerated sampling techniques (such as DDIM or DPMSolver) represent promising extensions.

---

### Chapter 7: Conclusion & References
- **Conclusion:** The project successfully demonstrates the theoretical and empirical principles of diffusion models through a complete, responsive web application and trained deep learning model.
- **Key References:**
  1. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
  2. Sohl-Dickstein, J., et al. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. ICML.
  3. Song, J., Meng, C., & Ermon, S. (2020). *Denoising Diffusion Implicit Models (DDIM)*. ICLR.
  4. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI.
