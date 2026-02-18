# Xception Architecture with Residual and SE Attention – CIFAR-10

This project documents my implementation and experimentation with advanced CNN architectures inspired by Xception and SE-Net. The work progresses from depthwise separable convolutions to residual connections, learning rate scheduling, and channel attention mechanisms.

All experiments are conducted on the CIFAR-10 dataset using PyTorch.

---
## Objective

The goals of this project were:

- Understand depthwise separable convolutions
- Implement residual connections manually
- Build a Mini Xception architecture from scratch
- Apply data augmentation for robustness
- Use cosine annealing learning rate scheduling
- Integrate Squeeze-and-Excitation (SE) attention
- Compare architectural improvements progressively

---

## Dataset – CIFAR-10

- 60,000 RGB images (32×32)
- 10 object classes
- 50,000 training images
- 10,000 test images

Loaded using `torchvision.datasets.CIFAR10`.

---

## Data Augmentation

To improve generalization and robustness:

- Random Horizontal Flip
- Random Rotation (10 degrees)
- Random Crop (32×32 with padding=4)
- Color Jitter (brightness, contrast, saturation, hue)
- Normalization (mean = 0.5, std = 0.5)

These transformations help the model learn invariant features.

---

# Part 1 – Depthwise Separable Convolution

## SeparableConv2d Implementation

Implemented custom depthwise separable convolution:

- Depthwise convolution (groups = in_channels)
- Pointwise convolution (1×1)

This separates:
- Spatial feature extraction
- Channel mixing

This reduces parameters compared to standard convolution.

---

# Part 2 – Mini Xception Architecture

## Architecture Overview

### Entry Flow
- Conv → BatchNorm → ReLU
- Conv → BatchNorm

### Middle Flow (Residual Blocks)
Three residual blocks built using:

- ReLU
- SeparableConv2d
- BatchNorm
- Skip connections

Blocks:
- 64 → 128 (stride=2)
- 128 → 256 (stride=2)
- 256 → 256 (stride=1)

### Exit Flow
- Final separable convolution
- BatchNorm
- Global Average Pooling
- Fully connected layer

---

## Residual Block Design

Each residual block:
- Main path: two separable convolutions
- Shortcut path: 1×1 convolution if dimensions change
- Output = main_path + shortcut

This improves gradient flow and stabilizes deep training.

---

## Training Setup (Initial Training)

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 1e-4
- Epochs: 25
- Batch size: 128

Observation:
The model stopped improving after around 15 epochs.

---

# Part 3 – Learning Rate Scheduling

To improve convergence:

- Implemented Cosine Annealing LR scheduler
- Scheduler gradually reduces learning rate
- Trained for 50 epochs

This allowed smoother optimization and improved stability.

---

# Part 4 – SE Attention Integration

## SEBlock Implementation

Implemented Squeeze-and-Excitation block:

1. Global Average Pooling (Squeeze)
2. Fully connected reduction layer
3. ReLU activation
4. Expansion layer
5. Sigmoid activation
6. Channel-wise reweighting

This allows the network to:
- Focus on important channels
- Suppress less informative features

---

# Part 5 – Mini Xception with SE (MiniXceptionSE)

Modified architecture:

- Entry convolution layers
- ResidualSEBlocks (with integrated SE attention)
- Channel expansion up to 728
- Global Average Pooling
- Dropout (0.4)
- Final classification layer

Each ResidualSEBlock:
- SeparableConv2d layers
- BatchNorm
- SE attention applied before residual addition
- Shortcut connection if dimensions change

---

## Final Training Setup

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 1e-4
- CosineAnnealingLR scheduler
- Epochs: 50

---

# Key Learnings

- Depthwise separable convolution reduces parameters while maintaining performance
- Residual connections stabilize deeper networks
- Learning rate scheduling improves convergence
- SE attention enhances channel-wise feature selection
- Global average pooling improves generalization
- Combining residual + separable conv + attention leads to stronger feature learning

---

# Conclusion

This project demonstrates a full architectural progression:

Standard CNN → Depthwise Separable CNN → Residual Xception → Learning Rate Scheduling → Xception with SE Attention.

The final model integrates:
- Efficient convolution (SeparableConv2d)
- Deep residual learning
- Attention mechanisms
- Modern optimization strategies

This work builds strong practical understanding of advanced CNN architecture design.

---
