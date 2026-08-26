# NewGAE Fix Implementation Plan

## Overview

This plan addresses the model collapse by fixing 5 root causes, improving the loss function for sparse adjacency, and properly handling variable-length graphs (1-15 nodes) within the fixed `max_nodes=15` framework.

---

## Fix #1: Replace Mean Pooling with Flatten in Encoder

**File:** [`src/models/NewGAE/Encoder.py`](src/models/NewGAE/Encoder.py)

**Problem:** `x.mean(dim=1)` collapses `(B, 15, 32)` → `(B, 32)`, losing all per-node structural information.

**Solution:** Flatten the node and feature dimensions instead:
```python
# Before:
x = x.mean(dim=1)  # (B, N, F) → (B, F) — LOSES INFORMATION

# After:
x = x.reshape(batch_size, -1)  # (B, N, F) → (B, N*F) — PRESERVES INFORMATION
```

**Impact:** `output_dim` changes from `dense_features[-1]` (e.g., 64) to `max_nodes * conv_channels[-1]` (e.g., 15*32=480). This means `fc_z` and `fc_mu`/`fc_logvar` input dimensions must be updated.

**Changes needed:**
- [`Encoder.py:68`](src/models/NewGAE/Encoder.py:68) — Replace `x.mean(dim=1)` with `x.reshape(batch_size, -1)`
- [`Encoder.py:60`](src/models/NewGAE/Encoder.py:60) — Update `self.output_dim` to `max_nodes * conv_channels[-1]` (or compute from actual flattened size)
- [`Encoder.py:46-58`](src/models/NewGAE/Encoder.py:46-58) — Remove the dense layers after convs (they now operate on flattened node features, which is a different paradigm) OR keep them but adjust input dim

**Design decision:** The dense layers after flattening in KlejdaGAE work on `(B, N*F)` which is fine. But in NewGAE, the dense layers were after mean pooling on `(B, F)`. With flatten, they'll operate on `(B, N*F)`. This is actually the same pattern as KlejdaGAE, so it should work. Just need to update `output_dim` to reflect the flattened size before dense layers.

---

## Fix #2: Add Node Positional Embedding in DecoderA

**File:** [`src/models/NewGAE/DecoderA.py`](src/models/NewGAE/DecoderA.py)

**Problem:** All 15 node embeddings are generated from the same latent vector via `view(B, 15, 15)`, so they tend to be identical → uniform adjacency.

**Solution:** Add a learnable positional embedding to differentiate nodes:
```python
self.node_pos_embed = nn.Parameter(torch.randn(1, max_nodes, self.node_embed_dim))

# In forward:
node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)
node_embeddings = node_embeddings + self.node_pos_embed  # differentiate nodes
adj_logits = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
```

**Changes needed:**
- [`DecoderA.py:35`](src/models/NewGAE/DecoderA.py:35) — Add `self.node_pos_embed` parameter after `self.node_generator`
- [`DecoderA.py:46`](src/models/NewGAE/DecoderA.py:46) — Add positional embedding to node_embeddings before bmm

---

## Fix #3: Weighted Loss for Sparse Adjacency

**File:** [`src/models/NewGAE/GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py) and [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py)

**Problem:** Adjacency matrix is ~95% zeros. BCEWithLogitsLoss treats all entries equally, so the model can achieve low loss by predicting "no edges everywhere".

**Solution A (Recommended):** Use `FocalLoss` — reduces loss for well-classified examples, focuses on hard positives:
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, preds, targets):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # probability of correct prediction
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**Solution B (Simpler):** Use weighted BCEWithLogitsLoss:
```python
# Compute pos_weight from data statistics
num_zeros = (adj == 0).sum()
num_ones = (adj == 1).sum()
pos_weight = torch.tensor([num_zeros / num_ones])  # e.g., ~19 for 5% density
self.criterion_a = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Changes needed:**
- Add FocalLoss implementation (or weighted BCE)
- Replace `self.criterion_a = nn.BCEWithLogitsLoss()` with the new loss

---

## Fix #4: Mask Padding Nodes in Loss Computation

**File:** [`src/models/NewGAE/GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py) and [`VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py)

**Problem:** Graphs have 1-15 nodes, but are zero-padded to 15. The loss penalizes the model for predicting non-zero features/edges for padding nodes, which is incorrect.

**Solution:** Create a mask from `parts_num` and apply it to the loss:
```python
def compute_reconstruction_loss(self, batch):
    x, adj, properties = batch
    parts_num = properties['parts_num']  # (B,) — number of real nodes per sample
    
    a_prime, x_prime, z = self.forward(x, adj)
    
    # Create node mask: (B, max_nodes, 1)
    node_mask = torch.arange(self.hparams.max_nodes, device=x.device).unsqueeze(0) < parts_num.unsqueeze(1)
    node_mask = node_mask.float()  # (B, max_nodes)
    
    # Adjacency mask: (B, max_nodes, max_nodes)
    adj_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
    
    # Feature mask: (B, max_nodes, 1) broadcast to (B, max_nodes, num_features)
    feat_mask = node_mask.unsqueeze(2)
    
    # Apply masks to loss computation
    loss_a = self.criterion_a(a_prime * adj_mask, adj * adj_mask)
    loss_x = self.criterion_x(x_prime * feat_mask, x * feat_mask)
    
    # Normalize by number of real elements
    loss_a = loss_a / (adj_mask.sum() + 1e-8)
    loss_x = loss_x / (feat_mask.sum() + 1e-8)
```

**Changes needed:**
- [`GraphAutoencoder.py:60-77`](src/models/NewGAE/GraphAutoencoder.py:60) — Add masking logic
- [`VariationalGraphAutoencoder.py:74-96`](src/models/NewGAE/VariationalGraphAutoencoder.py:74) — Add masking logic

---

## Fix #5: Condition DecoderX on Reconstructed Adjacency

**File:** [`src/models/NewGAE/DecoderX.py`](src/models/NewGAE/DecoderX.py)

**Problem:** DecoderX reconstructs features independently of the adjacency structure, so the model can learn features well while ignoring structure entirely.

**Solution:** Add conv layers that use `a_prime` as adjacency, similar to KlejdaGAE:
```python
def __init__(self, latent_dim, hidden_dims, max_nodes, num_features, config):
    # ... existing dense layers ...
    
    # Add conv layers that use a_prime
    self.convs = nn.ModuleList()
    current_channels = latent_dim  # or some intermediate dim
    for out_channels in config.get('decoder_x_conv_channels', [32, 16]):
        self.convs.append(ConvLayer(current_channels, out_channels, ...))
        current_channels = out_channels
    
    # Adjust final projection input dim

def forward(self, z, a_prime):
    # Expand z to node space
    x = self.z_to_nodes(z)  # (B, latent) → (B, latent*max_nodes)
    x = x.view(batch_size, self.max_nodes, -1)
    
    # Apply conv layers with a_prime as adjacency
    for conv in self.convs:
        x = conv(x, a_prime)
    
    # Flatten and apply dense layers
    x = x.reshape(batch_size, -1)
    for dense in self.denses:
        x = dense(x)
    
    x_prime = x.view(batch_size, self.max_nodes, self.num_features)
    return x_prime
```

**Changes needed:**
- [`DecoderX.py`](src/models/NewGAE/DecoderX.py) — Add conv layers, change forward signature to accept `a_prime`
- [`GraphAutoencoder.py:52-58`](src/models/NewGAE/GraphAutoencoder.py:52) — Pass `a_prime` to `decoder_x()`
- [`VariationalGraphAutoencoder.py:61-72`](src/models/NewGAE/VariationalGraphAutoencoder.py:61) — Pass `a_prime` to `decoder_x()`
- [`GraphAutoencoder.py:84-88`](src/models/NewGAE/GraphAutoencoder.py:84) — Fix `decode()` to pass `a_prime`
- [`VariationalGraphAutoencoder.py:106-110`](src/models/NewGAE/VariationalGraphAutoencoder.py:106) — Fix `decode()` to pass `a_prime`

---

## Fix #6: Fix `decode()` Return Order

**File:** [`src/models/NewGAE/GraphAutoencoder.py:84-88`](src/models/NewGAE/GraphAutoencoder.py:84) and [`VariationalGraphAutoencoder.py:106-110`](src/models/NewGAE/VariationalGraphAutoencoder.py:106)

**Problem:** `decode()` returns `(x_prime, a_prime)` but `forward()` returns `(a_prime, x_prime, z)`.

**Solution:** Change `decode()` to return `(a_prime, x_prime)` to match `forward()`:
```python
def decode(self, z: torch.Tensor):
    a_prime = self.decoder_a(z)
    x_prime = self.decoder_x(z, a_prime)  # pass a_prime
    return a_prime, x_prime  # ← FIXED ORDER
```

**Note:** The [`AutoencoderEvaluator.py:21`](src/deap/AutoencoderEvaluator.py:21) unpacks as `x_prime, a_prime = self.autoencoder.decode(latent_vector)`, so this call site must also be updated.

---

## Fix #7: Read `encoder_use_mlp` from Config

**File:** [`src/models/NewGAE/GraphAutoencoder.py:22`](src/models/NewGAE/GraphAutoencoder.py:22) and [`VariationalGraphAutoencoder.py:22`](src/models/NewGAE/VariationalGraphAutoencoder.py:22)

**Problem:** `use_mlp=True` is hardcoded instead of reading from config.

**Solution:**
```python
use_mlp=self.hparams.get('encoder_use_mlp', True),
```

---

## Fix #8: Uncomment `_init_weights` in VGAE

**File:** [`src/models/NewGAE/VariationalGraphAutoencoder.py:50`](src/models/NewGAE/VariationalGraphAutoencoder.py:50)

**Problem:** Weight initialization is commented out in VGAE but active in GAE.

**Solution:** Uncomment `self.apply(self._init_weights)`.

---

## Fix #9: Fix Float64 → Float32 Mismatch

**File:** [`utils/FramsticksGraphDataset.py`](utils/FramsticksGraphDataset.py)

**Problem:** Dataset returns `float64` tensors but model uses `float32`.

**Solution:** Cast to float32 in the dataset:
```python
x_matrix = torch.zeros((max_nodes, 5), dtype=torch.float32)
a_matrix = torch.zeros((max_nodes, max_nodes), dtype=torch.float32)
```

---

## Implementation Order

| Step | Fix | Files | Complexity |
|------|-----|-------|------------|
| 1 | Fix #9: Float64→Float32 | `FramsticksGraphDataset.py` | Trivial |
| 2 | Fix #7: encoder_use_mlp from config | `GraphAutoencoder.py`, `VariationalGraphAutoencoder.py` | Trivial |
| 3 | Fix #8: Uncomment _init_weights | `VariationalGraphAutoencoder.py` | Trivial |
| 4 | Fix #1: Replace mean pooling | `Encoder.py` | Medium |
| 5 | Fix #2: Node positional embedding | `DecoderA.py` | Easy |
| 6 | Fix #3: Weighted/Focal loss | `GraphAutoencoder.py`, `VariationalGraphAutoencoder.py` | Medium |
| 7 | Fix #4: Padding mask in loss | `GraphAutoencoder.py`, `VariationalGraphAutoencoder.py` | Medium |
| 8 | Fix #5: Condition DecoderX on a_prime | `DecoderX.py`, `GraphAutoencoder.py`, `VariationalGraphAutoencoder.py` | Complex |
| 9 | Fix #6: decode() return order | `GraphAutoencoder.py`, `VariationalGraphAutoencoder.py`, `AutoencoderEvaluator.py` | Easy |

---

## Files to Modify (Summary)

1. [`utils/FramsticksGraphDataset.py`](utils/FramsticksGraphDataset.py) — float64→float32
2. [`src/models/NewGAE/Encoder.py`](src/models/NewGAE/Encoder.py) — mean→flatten, update output_dim
3. [`src/models/NewGAE/DecoderA.py`](src/models/NewGAE/DecoderA.py) — add positional embedding
4. [`src/models/NewGAE/DecoderX.py`](src/models/NewGAE/DecoderX.py) — add conv layers, accept a_prime
5. [`src/models/NewGAE/GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py) — weighted loss, masking, pass a_prime, fix decode, use_mlp from config
6. [`src/models/NewGAE/VariationalGraphAutoencoder.py`](src/models/NewGAE/VariationalGraphAutoencoder.py) — weighted loss, masking, pass a_prime, fix decode, use_mlp from config, uncomment _init_weights
7. [`src/deap/AutoencoderEvaluator.py`](src/deap/AutoencoderEvaluator.py) — fix decode() unpack order