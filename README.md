# DenseNet-161 from Scratch

A from-scratch PyTorch implementation of DenseNet-161, trained on the full ImageNet-1k dataset. Built by following the architecture specifications in the original paper:

> **Densely Connected Convolutional Networks**  
> Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger  
> CVPR 2017 — [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)

---

## Architecture

DenseNet replaces the standard residual connection (add) with dense connections (concatenate): every layer receives feature maps from **all preceding layers** in its block, and passes its own to all subsequent ones. This maximizes feature reuse, reduces the number of parameters, and provides strong gradient flow during training.

### DenseNet-161 Configuration

| Component | Detail |
|---|---|
| Growth rate `k` | 48 |
| Dense blocks | 4 |
| Layers per block | 6 — 12 — 36 — 24 |
| Bottleneck | BN → ReLU → 1×1 Conv (4k) → BN → ReLU → 3×3 Conv (k) |
| Transition | 1×1 Conv (compression θ=0.5) → 2×2 AvgPool |
| Initial stem | 7×7 Conv (96 filters, stride 2) → BN → ReLU → 3×3 MaxPool (stride 2) |
| Classifier | AdaptiveAvgPool → Flatten → Linear(2208 → 1000) |

### Channel progression

```
Stem output:       96
After Dense-1:     96  + 6×48  = 384
After Transition-1:               192
After Dense-2:     192 + 12×48 = 768
After Transition-2:               384
After Dense-3:     384 + 36×48 = 2112
After Transition-3:               1056
After Dense-4:     1056 + 24×48 = 2208
```

---

## Project Structure

```
DenseNet161/
├── build.ipynb          # Model definition + training loop (Google Colab)
├── evaluation.ipynb     # Top-1 / Top-5 accuracy on ImageNet validation set
├── densenet161_checkpoint.pt   # Saved model checkpoint
└── imagenet_500/        # Local 500-image evaluation cache
```

---

## Training

Training was run on Google Colab with a GPU. The full ImageNet-1k training split (~1.28M images) was loaded in **streaming mode** from Hugging Face to avoid downloading the entire dataset locally.

- **Optimizer:** Adam (lr=1e-3)
- **Loss:** CrossEntropyLoss
- **Batch size:** 64
- **Epochs:** 3
- **Preprocessing:** Resize 256 → CenterCrop 224 → Normalize (ImageNet mean/std)
- **Checkpointing:** Saved to Google Drive every 500 batches

---

## Evaluation

Evaluated on the full ImageNet-1k validation set (50,000 images) using the saved checkpoint.

```python
# Run evaluation.ipynb in Colab to reproduce
```

Metrics reported: **Top-1 accuracy** and **Top-5 accuracy**.

---

## Requirements

```
torch
torchvision
datasets        # Hugging Face datasets
huggingface_hub
tqdm
```

---

## References

- Huang et al., *Densely Connected Convolutional Networks*, CVPR 2017
