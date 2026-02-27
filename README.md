# Zero-DCE Low-Light Image Enhancement

Self-supervised low-light image enhancement using the Zero-DCE (CVPR 2020) framework.  
This project replicates and evaluates Zero-Reference Deep Curve Estimation (Zero-DCE) for enhancing low-light images without paired supervision.

---
Project executed by : Anish A. Kale, Aryan Ramchandra, Adolfo Bugarin - UC Riverside, Riverside, CA
---

## Overview

Low-light image enhancement is critical for improving downstream computer vision tasks such as object detection and surveillance in real-world environments.

Instead of directly predicting a brightened image, Zero-DCE learns pixel-wise iterative enhancement curves:

I_enhanced = I + α · I · (1 − I)

The model predicts 8 curve maps per RGB channel and applies them iteratively to progressively improve illumination while preserving structural details.

---

## Architecture

- 7-layer convolutional network (DCE-Net)
- Skip connections
- 24 output channels (8 curve maps × RGB)
- Iterative curve-based enhancement
- No pooling (lightweight design)

---

## Loss Design (Self-Supervised)

Training does not require paired ground-truth images. The model is optimized using:

- Exposure Control Loss
- Color Constancy Loss
- Spatial Consistency Loss
- Illumination Smoothness (Total Variation) Loss

This enables training directly on low-light datasets.

---

## Training Details

- Dataset size: ~2000 images
- Optimizer: Adam
- Training epochs: 200
- Gradient clipping applied
- Stable convergence observed after ~80–100 epochs

Cross-dataset evaluation performed on:
- LOL
- LOL-V2
- DarkFace

Reported PSNR: ~27.9 dB (comparable to original paper)

---

## How to Train

```bash
python src/lowlight_train.py
```

---

## How to Run Inference

```bash
python src/test.py
```

---

## Project Context

This project was developed as part of a graduate-level computer vision course to replicate and evaluate a CVPR 2020 paper.  
The goal was to understand self-supervised loss design and curve-based image enhancement mechanisms.

---

## Future Improvements

- Add learning rate scheduling
- Include validation monitoring during training
- Compare against GAN-based enhancement models
- Evaluate impact on downstream object detection

---

## License

MIT License
