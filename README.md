# 🧠 Reproduction and Analysis of Self-Supervised Poisson Denoising: The Poisson2Sparse Framework

> **Bachelor's Thesis — Marmara University, Electrical & Electronics Engineering**  
> Supervisor: Asst. Prof. Dr. Rıfat Volkan Şenyuva | January 2026

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA-orange?logo=pytorch)](https://pytorch.org)
[![Apple Silicon](https://img.shields.io/badge/Hardware-Apple%20M4-black?logo=apple)](https://apple.com)
[![PSNR](https://img.shields.io/badge/PSNR-31.88%20dB-brightgreen)]()

---

## 📌 Overview

Biomedical imaging techniques such as **fluorescence microscopy** and **MRI** are heavily affected by Poisson noise — especially in low-signal conditions. Unlike Gaussian noise, Poisson noise is signal-dependent and cannot be removed with standard supervised denoising methods, since clean reference images are rarely available in clinical settings.

This thesis provides a **detailed analysis and independent reproduction** of the **Poisson2Sparse** framework — a self-supervised denoising method that learns entirely from noisy data, with no clean ground-truth required.

---

## 🔬 What is Poisson2Sparse?

Poisson2Sparse combines two powerful ideas:

1. **Convolutional Sparse Coding (CSC)** — represents an image as a sparse linear combination of learned filters, capturing the underlying anatomical structure while separating out noise.
2. **Deep Algorithm Unrolling (LISTA)** — converts the iterative ISTA optimization algorithm into a trainable neural network, so the model learns optimal parameters directly from data.

Together, they form a compact neural network that is both mathematically principled and practical to train — even without a GPU.

---

## 🎯 Core Contributions

- **Independent full reimplementation** of the Poisson2Sparse framework from scratch
- **Cross-platform hardware adaptation**: ported CUDA-based training to Apple Silicon MPS (Metal Performance Shaders)
- **Reproducibility study**: verified that self-supervised deep denoising works on consumer-grade hardware — no research lab required
- **Robustness analysis**: evaluated model behavior under varying noise levels and architectures

---

## 📊 Results

| Metric | This Work | Original Paper |
|--------|-----------|----------------|
| PSNR (dB) | **31.88** | 32.20 |
| Dataset | PINCAT | PINCAT |
| Hardware | Apple M4 (MPS) | CUDA GPU |
| Training | Self-supervised | Self-supervised |

> The model trained on Apple M4 reached **31.88 dB PSNR**, closely approaching the original paper's 32.20 dB benchmark — demonstrating that high-performance denoising is achievable on consumer hardware.

Method noise maps confirmed that the network removes random Poisson noise while preserving underlying anatomical structures intact.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| PyTorch (MPS backend) | Model training on Apple Silicon |
| NumPy | Numerical operations |
| Matplotlib | Visualization of denoising results |
| PINCAT Dataset | Cardiac imaging benchmark dataset |

---

## 📁 Repository Structure

```
thesis-image-processing/
│
├── model/
│   ├── poisson2sparse.py      # Core network (CSC + LISTA unrolling)
│   ├── lista_layers.py        # Deep unrolling layers
│   └── loss.py                # Poisson log-likelihood loss
│
├── data/
│   └── pincat/                # PINCAT dataset loader
│
├── train.py                   # Training script (MPS/CUDA compatible)
├── evaluate.py                # PSNR & SSIM evaluation
├── visualize.py               # Denoising output visualization
│
├── EE4297_Report.pdf          # Full thesis report
└── README.md
```

---

## 🧮 Key Concepts

**Poisson Noise Model**

Unlike additive Gaussian noise, Poisson noise scales with signal intensity. The log-likelihood loss used for training is:

```
L(θ) = Σ [ f_θ(y) - y · log(f_θ(y)) ]
```

where `y` is the noisy observation and `f_θ` is the network output.

**Deep Unrolling (LISTA)**

Each layer of the network corresponds to one iteration of the ISTA algorithm:

```
z^(k+1) = Soft_τ( W_e · x + S · z^(k) )
x_hat   = W_d · z^(K)
```

Parameters `W_e`, `S`, `W_d`, and `τ` are all learned end-to-end.

---

## ▶️ How to Run

```bash
# Clone the repository
git clone https://github.com/abdullahaltepe-hue/thesis-image-processing.git
cd thesis-image-processing

# Install dependencies
pip install torch numpy matplotlib

# Train the model (Apple Silicon)
python train.py --device mps --epochs 100

# Evaluate on PINCAT
python evaluate.py --checkpoint checkpoints/best_model.pth
```

---

## 📄 Full Report

The complete academic report is available: [`EE4297_Report.pdf`](./EE4297_Report.pdf)

**Keywords:** Self-Supervised Learning · Image Denoising · Poisson Noise · Sparse Coding · Deep Unrolling · Biomedical Imaging · Reproducibility

---

## 👤 Author

**Abdullah Altepe** — Marmara University, EEE (Student ID: 150721034)  
[GitHub](https://github.com/abdullahaltepe-hue)

*Submitted in partial fulfillment of the requirements for BSc in Electrical and Electronics Engineering, Marmara University, January 2026.*
