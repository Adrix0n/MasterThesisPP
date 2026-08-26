# NewGAE Model Structure Review

## Overview

I've reviewed all 7 files in `src/models/NewGAE/` against:
- The base class [`src/models/BaseGraphAutoEncoder.py`](src/models/BaseGraphAutoEncoder.py)
- The reference implementation [`src/models/KlejdaGAE/`](src/models/KlejdaGAE/)
- The config file [`configs/gae_config.yaml`](configs/gae_config.yaml)

---

## Issues Found

### 🔴 CRITICAL: Inconsistent return order in `decode()` vs `forward()`

**Files:** [`GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py:84), [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py:106)

- `forward()` returns `(a_prime, x_prime, z)` — adjacency first, features second
- `decode()` returns `(x_prime, a_prime)` — features first, adjacency second

This is a **bug** waiting to happen. Any code calling `decode()` and unpacking as `(a_prime, x_prime)` will silently get swapped tensors.

---

### 🔴 CRITICAL: Config parameter `decoder_x_conv_channels` is unused

**Config:** [`gae_config.yaml`](configs/gae_config.yaml:38) defines `decoder_x_conv_channels: [32, 16]`

**NewGAE DecoderX:** [`DecoderX.py`](src/models/NewGAE/DecoderX.py) is a **pure MLP decoder** — it has no convolutional layers at all. The config parameter `decoder_x_conv_channels` is completely ignored.

Compare with [`KlejdaDecoderX`](src/models/KlejdaGAE/KlejdaDecoderX.py) which uses conv layers with the reconstructed adjacency matrix `a_prime`.

---

### 🟡 WARNING: Weight initialization commented out in VGAE

**File:** [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py:50)

```python
# self.apply(self._init_weights)
```

The [`GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py:49) calls `self.apply(self._init_weights)`, but the VGAE variant has it commented out. This inconsistency means the VGAE uses default PyTorch initialization while GAE uses Kaiming initialization.

---

### 🟡 WARNING: `encoder_use_mlp` config parameter is hardcoded

**Config:** [`gae_config.yaml`](configs/gae_config.yaml:18) defines `encoder_use_mlp: true`

**Code:** Both [`GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py:22) and [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py:22) hardcode `use_mlp=True` instead of reading `self.hparams.encoder_use_mlp`.

The config value has no effect — you cannot disable the MLP projection without editing code.

---

### 🟡 WARNING: Unused imports

**Files:** [`GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py:3-4), [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py:3-4)

- `import pytorch_lightning as pl` — unused (inherited from base)
- `import torch.optim as optim` — unused (optimizer configured in base class)

---

### 🟡 WARNING: Missing `__init__.py` — no external integration

**Directory:** [`src/models/NewGAE/`](src/models/NewGAE/)

- No `__init__.py` exists (confirmed by search — no `__init__.py` anywhere in `src/models/`)
- No file outside `NewGAE/` imports from this module
- The model is **not integrated** into any training pipeline, evaluator, or experiment script

---

### ℹ️ INFO: Architectural differences from KlejdaGAE (intentional design changes)

| Aspect | KlejdaGAE | NewGAE | Notes |
|--------|-----------|--------|-------|
| **Encoder pooling** | Flatten `(B, N*F)` | Mean pool `mean(dim=1)` | NewGAE produces graph-level representation |
| **DecoderX structure** | Conv + Dense layers | Pure Dense/MLP layers | NewGAE is simpler, no conv in DecoderX |
| **DecoderX input** | `(z, a_prime)` — uses reconstructed adj | `(z)` — only latent vector | NewGAE DecoderX is independent of DecoderA |
| **Adjacency loss** | `MSELoss` + Sigmoid | `BCEWithLogitsLoss` (no sigmoid) | NewGAE uses proper binary classification loss |
| **Feature loss** | `MSELoss` | `HuberLoss` | NewGAE uses more robust loss |
| **Normalization** | `LayerNorm` | `BatchNorm1d` | Different normalization strategy |
| **DecoderA output** | Sigmoid → `(0,1)` | Raw logits (no activation) | Consistent with BCEWithLogitsLoss |

These are **intentional design improvements**, not bugs.

---

### ℹ️ INFO: Minor observations

1. [`DecoderA.py`](src/models/NewGAE/DecoderA.py:4) imports `sqrt` from `numpy` — works but `math.sqrt` would be more lightweight
2. [`Encoder.py`](src/models/NewGAE/Encoder.py:20) stores `self.max_nodes` but never uses it in forward — harmless
3. [`Encoder.py`](src/models/NewGAE/Encoder.py:21) stores `self.use_mlp` but never uses it outside `__init__` — harmless

---

## Summary

| Severity | Count | Action Required |
|----------|-------|-----------------|
| 🔴 Critical | 2 | Fix `decode()` return order; address unused `decoder_x_conv_channels` |
| 🟡 Warning | 4 | Uncomment `_init_weights` in VGAE; read `encoder_use_mlp` from config; remove unused imports; add `__init__.py` |
| ℹ️ Info | 5 | No action needed — intentional design differences |

The most impactful bug is the **inconsistent return order in `decode()`** — this will cause silent data corruption if `decode()` is ever called directly.