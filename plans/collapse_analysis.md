# Model Collapse Analysis — NewGAE

## End-to-End Tensor Shape Trace

### Input (from Dataset)
| Tensor | Shape | dtype | Description |
|--------|-------|-------|-------------|
| `x` | `(B, 15, 5)` | float64 | Node features [x, y, z, fr, ing] |
| `adj` | `(B, 15, 15)` | float64 | Adjacency matrix (sparse, mostly 0s) |

### Encoder forward
```
x: (B, 15, 5) ──► MLP Linear(5→32) ──► (B, 15, 32)
                                              │
                                    ConvLayer [16, 32]:
                                    ┌─────────────────────────┐
                                    │ conv(x, adj) → (B,N,F)  │
                                    │ transpose(1,2) → (B,F,N) │
                                    │ BatchNorm1d(F) ✓         │
                                    │ transpose(1,2) → (B,N,F) │
                                    │ act + dropout             │
                                    └─────────────────────────┘
                                              │
                                    (B, 15, 32)
                                              │
                                    mean(dim=1) ← ★ CRITICAL ★
                                              │
                                    (B, 32)
                                              │
                                    Dense(32→128) → Dense(128→64)
                                              │
                                    (B, 64) = hidden_features
```

### Latent projection
```
hidden_features: (B, 64) ──► fc_z: Linear(64→15) ──► z: (B, 15)
```

### DecoderA forward
```
z: (B, 15) ──► Dense(15→64) → Dense(64→128) ──► (B, 128)
                                                       │
              node_generator: Linear(128→15*15=225) ──► (B, 225)
                                                       │
              view(B, 15, 15) ──► node_embeddings: (B, 15, 15)
                                                       │
              bmm(emb, emb^T) ──► (B, 15, 15)
                                                       │
              / sqrt(15) ──► a_prime: (B, 15, 15) ← raw logits
```

### DecoderX forward
```
z: (B, 15) ──► Dense(15→64) → Dense(64→128) ──► (B, 128)
                                                       │
              final_projection: Linear(128→15*5=75) ──► (B, 75)
                                                       │
              view(B, 15, 5) ──► x_prime: (B, 15, 5)
```

### Loss computation
```
loss_a = BCEWithLogitsLoss(a_prime, adj)    # a_prime: (B,15,15), adj: (B,15,15) ✓
loss_x = HuberLoss(x_prime, x)              # x_prime: (B,15,5),  x: (B,15,5)   ✓

recon_loss = (500.0 * loss_a) + loss_x
```

---

## Root Cause Analysis: Why the Model Collapses

### 🔴 CAUSE #1 (Most Critical): Mean Pooling Destroys Structural Information

**File:** [`Encoder.py`](src/models/NewGAE/Encoder.py:68)

```python
x = x.mean(dim=1)  # (B, 15, 32) → (B, 32)
```

This is the **primary cause of collapse**. Here's why:

| Aspect | KlejdaGAE (works) | NewGAE (collapses) |
|--------|-------------------|-------------------|
| After convs | `(B, 15, 32)` | `(B, 15, 32)` |
| Pooling | `view(B, -1)` → `(B, 480)` | `mean(dim=1)` → `(B, 32)` |
| Information retained | All 480 values preserved | Only 32 averages kept |
| Node count info | Preserved in vector length | **Lost** — 3 nodes with mean=5 and 15 nodes with mean=5 are identical |
| Node differentiation | Preserved | **Lost** — all nodes averaged together |

**Concrete example of the problem:**
- Graph A: 3 nodes with features `[[10,0,0], [0,10,0], [0,0,10]]` → mean = `[3.3, 3.3, 3.3]`
- Graph B: 15 nodes all at `[3.3, 3.3, 3.3]` → mean = `[3.3, 3.3, 3.3]`
- **The encoder produces the exact same latent vector for both!**

The decoder then has no information to reconstruct different graphs, so it collapses to a single "average" output.

### 🔴 CAUSE #2: DecoderA Generates All Node Embeddings from the Same Latent

**File:** [`DecoderA.py`](src/models/NewGAE/DecoderA.py:35-47)

```python
self.node_generator = nn.Linear(current_in_features, max_nodes * self.node_embed_dim)
# ...
node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)
adj_logits = torch.bmm(node_embeddings, node_embeddings.transpose(1, 2))
```

All 15 node embeddings are generated from the **same** latent vector `z: (B, 15)`. The `Linear(128→225)` layer produces 225 values that are reshaped into 15×15. But without any mechanism to differentiate nodes (like positional encoding or a sequential generation process), the model tends to produce **nearly identical embeddings** for all nodes.

When all node embeddings are identical, `bmm(emb, emb^T)` produces a matrix where all entries are the same value — the model cannot represent different edge patterns.

### 🟡 CAUSE #3: Sparse Adjacency + BCEWithLogitsLoss = Gradient Vanishing

**File:** [`GraphAutoencoder.py`](src/models/NewGAE/GraphAutoencoder.py:46)

```python
self.criterion_a = nn.BCEWithLogitsLoss()
```

The adjacency matrix is very sparse (most entries = 0, few = 1). BCEWithLogitsLoss on a highly imbalanced target:
- If the model predicts all logits → -∞ (all zeros after sigmoid), loss ≈ 0 for zero entries
- The model can achieve **low loss by predicting no edges anywhere**
- This reinforces the collapse: no edges → simple structure → easy to reconstruct with mean-pooled latent

### 🟡 CAUSE #4: DecoderX is Independent of Graph Structure

**File:** [`DecoderX.py`](src/models/NewGAE/DecoderX.py:38-48)

```python
def forward(self, z: torch.Tensor) -> torch.Tensor:
    # ... pure MLP, no adjacency input
    x_prime = x.view(batch_size, self.max_nodes, self.num_features)
    return x_prime
```

Compare with [`KlejdaDecoderX`](src/models/KlejdaGAE/KlejdaDecoderX.py:57-74):
```python
def forward(self, z: torch.Tensor, a_prime: torch.Tensor) -> torch.Tensor:
    # ... uses a_prime as adjacency for conv layers
```

In KlejdaGAE, the feature decoder is conditioned on the reconstructed adjacency matrix, creating a **coordinated reconstruction**. In NewGAE, features and adjacency are reconstructed independently, so the model can learn to reconstruct features well while ignoring structure entirely.

### ℹ️ CAUSE #5: Float64 vs Float32 Mismatch

**Dataset:** Returns `float64` tensors
**Model:** All `nn.Linear` layers use `float32` (PyTorch default)

When a `float64` tensor is passed through a `float32` Linear layer, PyTorch will raise a runtime error OR silently cast. This can cause numerical instability.

---

## Summary: Why It Collapses

```
Mean pooling (B,15,32)→(B,32)
        │
        ▼
Encoder loses all per-node information
        │
        ▼
Latent z cannot distinguish different graphs
        │
        ▼
DecoderA generates uniform node embeddings → uniform adjacency
DecoderX generates features independent of structure
        │
        ▼
BCEWithLogitsLoss on sparse adj rewards "all zeros"
        │
        ▼
MODEL COLLAPSE — all outputs converge to the same "average" graph
```

## Recommended Fixes

### Fix #1 (Critical): Replace mean pooling with flatten
In [`Encoder.py`](src/models/NewGAE/Encoder.py:68), change:
```python
x = x.mean(dim=1)  # (B, N, F) → (B, F) — LOSES INFORMATION
```
to:
```python
x = x.reshape(batch_size, -1)  # (B, N, F) → (B, N*F) — PRESERVES INFORMATION
```

### Fix #2 (Critical): Add node differentiation in DecoderA
In [`DecoderA.py`](src/models/NewGAE/DecoderA.py), add a learned node embedding or positional encoding so each of the 15 nodes gets a different starting point:
```python
self.node_pos_embed = nn.Parameter(torch.randn(1, max_nodes, node_embed_dim))
# In forward:
node_embeddings = x.view(batch_size, self.max_nodes, self.node_embed_dim)
node_embeddings = node_embeddings + self.node_pos_embed  # differentiate nodes
```

### Fix #3 (Important): Condition DecoderX on adjacency
In [`DecoderX.py`](src/models/NewGAE/DecoderX.py), pass `a_prime` as input and use conv layers like KlejdaGAE does.

### Fix #4 (Important): Handle adjacency sparsity
Use weighted BCE loss or focal loss to prevent the model from collapsing to "all zeros":
```python
pos_weight = torch.tensor([num_zeros / num_ones])  # weight positive edges higher
self.criterion_a = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

### Fix #5 (Good practice): Fix dtype consistency
Cast dataset tensors to float32 before feeding to the model.