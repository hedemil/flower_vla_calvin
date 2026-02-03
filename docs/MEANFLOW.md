
# Mean Flow Decoder Integration Guide

This document outlines the steps needed to integrate the Mean Flow decoder from JAX/Flax into the FLOWER VLA PyTorch architecture.

## ⚠️ IMPORTANT UPDATE (2026-01-27)

**CRITICAL CORRECTION**: This guide has been updated to use the correct automatic differentiation method.

- **Original (INCORRECT)**: Used `torch.autograd.grad` (backward-mode AD) ❌
- **Corrected (CORRECT)**: Uses `torch.func.jvp` (forward-mode AD) ✅

The correction is based on the [official PyTorch implementation](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py). See Section 5 and the References section at the end for details.

---

## Overview of Mean Flow

Mean Flow differs from standard Rectified Flow in that it:
1. **Uses average velocity** `u` over timestep interval `[r, t]` instead of instantaneous velocity
2. **Requires timestep difference** `h = t - r` as input to the decoder
3. **Computes loss using automatic differentiation** to get `du/dt` via JVP (Jacobian-vector product)
4. **Uses two timesteps** `t` and `r` sampled with constraints: `t >= r`, and for a proportion of samples `r = t`

---

## Step-by-Step Integration

### 1. Create Mean Flow Decoder Architecture

**Location:** Create new module in `flower/models/networks/transformers.py` or inline in `flower.py`

**Action:**
Create a decoder that takes timestep difference `h` as additional input.

**Note:** In the official PyTorch implementation, the network receives `(t, h)` as conditioning, where both are embedded. In FLOWER, we already embed `t` in the DiT, so the decoder only needs to embed `h`.

```python
class MeanFlowDecoder(nn.Module):
    def __init__(self, dit_dim: int, action_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or dit_dim * 2

        # Time embedding for h (timestep difference)
        # Reuse TimestepEmbedder architecture (sinusoidal embeddings)
        self.h_embedder = TimestepEmbedder(dit_dim)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(dit_dim * 2, hidden_dim),  # dit_dim (features) + dit_dim (h_embed)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent features [B, T, dit_dim]
            h: Timestep difference (t - r) [B] or [B, 1, 1]
        Returns:
            Average velocity u [B, T, action_dim]
        """
        # Flatten h to [B] for embedding
        h_flat = h.view(-1)  # [B]

        # Embed timestep difference
        h_embed = self.h_embedder(h_flat)  # [B, dit_dim]
        h_embed = h_embed.unsqueeze(1).expand(-1, z.shape[1], -1)  # [B, T, dit_dim]

        # Concatenate and decode
        z_h = torch.cat([z, h_embed], dim=-1)  # [B, T, dit_dim*2]
        return self.decoder(z_h)
```

**Alternative:** If you want to match the official implementation more closely, you could condition on both `t` and `h` in the DiT blocks themselves, rather than just in the decoder.

---

### 2. Update Decoder Initialization

**Location:** `_setup_dit_components` (lines 374-379)

**Current:**
```python
self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)
```

**New:**
```python
self.action_decoders[action_name] = MeanFlowDecoder(
    dit_dim=dit_dim,
    action_dim=input_dim,
    hidden_dim=dit_dim * 2
).to(self.device)
```

---

### 3. Update `decode_actions` to Accept Timestep Difference

**Location:** `decode_actions` (lines 857-873)

**Current Signature:**
```python
def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
```

**New Signature:**
```python
def decode_actions(self, z: torch.Tensor, h: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
    """
    Args:
        z: Latent features [B, T, dit_dim]
        h: Timestep difference (t - r) [B]
        action_type: Action type indices [B, T, action_dim]
        valid_dims: Valid dimensions mask [B, T, action_dim]
    """
    default_dtype = next(self.parameters()).dtype
    batch_size = z.shape[0]
    max_action_dim = self.action_dim
    decoded = torch.zeros(batch_size, z.shape[1], max_action_dim,
                    device=self.device).to(default_dtype)

    for action_name, action_idx in self.action_space_index.action_spaces.items():
        mask = (action_type == action_idx)
        if mask.any():
            # Pass h to the decoder
            pred = self.action_decoders[action_name](z, h)
            decoded = pred
    return decoded
```

---

### 4. Update `dit_forward` to Pass Timestep Difference

**Location:** `dit_forward` (lines 570-632)

**Current Signature (line 570):**
```python
def dit_forward(self, z: torch.Tensor, t: torch.Tensor, cond_dict: dict) -> torch.Tensor:
```

**Current Return (line 632):**
```python
return self.decode_actions(cx, action_type, valid_dims)
```

**Changes Needed:**

1. **Add `h` parameter** to function signature:
```python
def dit_forward(self, z: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond_dict: dict) -> torch.Tensor:
    """
    Forward pass through the DiT blocks.

    Args:
        z: Latent actions [B, T, action_dim]
        t: Current timestep [B] or [B, 1, 1]
        h: Timestep difference (t - r) [B] or [B, 1, 1]
        cond_dict: Conditioning dictionary
    """
```

2. **Pass `h` to decoder** (line 632):
```python
return self.decode_actions(cx, h, action_type, valid_dims)
```

**Note:** The `h` parameter flows through without modification until it reaches the decoder.

---

### 5. Implement Mean Flow Loss with JVP (Jacobian-Vector Product)

**Location:** Add new method or replace `rf_loss` (lines 497-546)

**CRITICAL:** Use `torch.func.jvp` for forward-mode automatic differentiation (NOT `torch.autograd.grad`)

**Reference:** [Official PyTorch Implementation](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py)

**New Method:**
```python
def meanflow_loss(self, cond, actions, dataset_idx=None):
    """
    Compute the mean flow loss using JVP (Jacobian-vector product).

    Based on official PyTorch implementation:
    https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py
    """
    default_dtype = next(self.parameters()).dtype

    if len(actions.shape) == 4:
        actions = actions.squeeze(1)
    b = actions.size(0)
    device = actions.device
    actions = actions.to(default_dtype)

    # Sample t and r with constraint t >= r
    t, r = self.sample_tr(b)

    # Interpolate between actions and noise: z_t = (1-t)*x + t*e
    texp = t.view([b] + [1] * (actions.dim() - 1))
    rexp = r.view([b] + [1] * (actions.dim() - 1))

    e = torch.randn_like(actions, device=device).to(default_dtype)
    z = (1 - texp) * actions + texp * e

    # Instantaneous velocity
    v = e - actions

    # Optional: Guidance velocity (implement if using CFG)
    # v_g = self.guidance_fn(v, z, t, cond)
    # y_inp, v_g = self.cond_drop(v, v_g, cond)

    # -------------------------------------------------------------------
    # Define network function for JVP
    # -------------------------------------------------------------------
    def u_func(z_input, t_input, r_input):
        """Network function that computes u(z, t, h=t-r)"""
        h_input = t_input - r_input
        return self.dit_forward(z_input, t_input, h_input, cond)

    # Define tangent vectors for JVP
    # JVP computes: du/dz * v + du/dt * dtdt + du/dr * drdt
    dtdt = torch.ones_like(t).view([b] + [1] * (actions.dim() - 1))
    drdt = torch.zeros_like(r).view([b] + [1] * (actions.dim() - 1))

    # -------------------------------------------------------------------
    # Compute u and du/dt using JVP (single forward pass!)
    # -------------------------------------------------------------------
    with torch.amp.autocast("cuda", enabled=False):
        # JVP returns: (u, du/dz * v + du/dt * dtdt + du/dr * drdt)
        # Since drdt=0, we get: (u, du/dz * v + du/dt)
        u_pred, dudt_combined = torch.func.jvp(
            u_func,
            (z, texp, rexp),  # Primals (inputs)
            (v, dtdt, drdt)   # Tangents (directions)
        )

        # dudt_combined = du/dz * v + du/dt * 1
        # We need just du/dt, so run JVP again with v=0
        _, dudt = torch.func.jvp(
            u_func,
            (z, texp, rexp),
            (torch.zeros_like(v), dtdt, drdt)
        )

        # Compute target average velocity
        # u_tgt = v - h * du/dt
        h = (texp - rexp).clamp(min=0.0, max=1.0)
        u_tgt = (v - h * dudt).detach()

        # Compute loss (squared L2)
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=(1, 2))  # Sum over sequence and action dims

        # Adaptive weighting (optional but recommended)
        norm_eps = 0.01
        norm_p = 1.0
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = loss / adp_wt

        loss = loss.mean()  # Mean over batch

    # Monitoring metrics
    with torch.no_grad():
        v_loss = ((u_pred - v) ** 2).mean()

    losses_dict = {
        "loss": loss.item(),
        "v_loss": v_loss.item(),
        "h_mean": h.mean().item(),
    }

    return loss, losses_dict
```

**Key Points:**
1. **JVP not backward AD**: Uses `torch.func.jvp` which is forward-mode automatic differentiation
2. **Two JVP calls**: First gets `u_pred` and combined derivative, second gets pure `du/dt`
3. **Tangent vectors**: `(v, dtdt=1, drdt=0)` for the directional derivative
4. **Adaptive weighting**: Balances training across different loss magnitudes
5. **Mixed precision**: Explicitly disables autocast for JVP computation

---

### 6. Update `sample_tr` Method

**Location:** Already exists (lines 590-604), but needs modification

**Current Implementation:**
```python
def sample_tr(self, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample time t and compute scaling factor r(t).
    """
    t = self.noise_dist(b)
    r = self.noise_dist(b)

    t, r = torch.maximum(t, r), torch.minimum(t, r)

    return t, r
```

**Updated Implementation:**
```python
def sample_tr(self, b: int, data_proportion: float = 0.75) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample timesteps t and r with constraint t >= r.
    For data_proportion of samples, set r = t (instantaneous velocity).

    Args:
        b: Batch size
        data_proportion: Fraction of batch where r = t

    Returns:
        t: End timestep [b]
        r: Start timestep [b]
    """
    t = self.noise_dist(b)
    r = self.noise_dist(b)

    # Ensure t >= r
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # For data_proportion of samples, set r = t
    data_size = int(b * data_proportion)
    zero_mask = torch.arange(b, device=t.device) < data_size
    zero_mask = zero_mask.view(b, 1)
    r = torch.where(zero_mask, t, r)

    return t, r
```

**Add to `__init__`:**
```python
self.data_proportion = 0.75  # Proportion of samples with r=t
```

---

### 7. Update Training Step to Use Mean Flow Loss

**Location:** `training_step` (lines 435-473)

**Current (line 448):**
```python
act_loss, losses_dict = self.rf_loss(obs_features, dataset_batch["actions"])
```

**New:**
```python
act_loss, losses_dict = self.meanflow_loss(obs_features, dataset_batch["actions"])
```

---

### 8. Update Sampling to Use Timestep Difference

**Location:** `sample_actions` (lines 548-568)

**Current (line 565):**
```python
vc = self.dit_forward(z, t_tensor, cond)
```

**Updated:**
```python
# For Mean Flow sampling: u(z_t, t, h=t-r) where r is the next timestep
# In the official implementation, single-step sampling uses h = t (from t to 0)
r_val = (i - 1) / steps  # Next timestep
r_tensor = torch.full((b,), r_val, device=device)
h_tensor = t_tensor - r_tensor  # Timestep difference

u = self.dit_forward(z, t_tensor, h_tensor, cond)
z = z - h_tensor.view([b] + [1]*(z.dim()-1)) * u
```

**Alternative (Single-step inference):**
```python
# For 1-step generation (as in official implementation):
# z_1 -> z_0 using u(z_1, t=1, h=1)
if inference and steps == 1:
    t_tensor = torch.ones((b,), device=device)
    h_tensor = torch.ones((b,), device=device)  # h = t - r = 1 - 0
    u = self.dit_forward(z, t_tensor, h_tensor, cond)
    z = z - u  # z_0 = z_1 - u
    return z.clamp(-1, 1)
```

**Note:** The official PyTorch implementation uses 1-step sampling by default, where `h = t = 1` to go directly from noise to data.

---

## Summary of Changes

| Location | Change | Complexity |
|----------|--------|------------|
| 1. Create `MeanFlowDecoder` | New decoder architecture with h embedding | Medium |
| 2. `_setup_dit_components:379` | Replace `nn.Linear` with `MeanFlowDecoder` | Easy |
| 3. `decode_actions:782-798` | Add `h` parameter, pass to decoder | Easy |
| 4. `dit_forward:570,632` | Add `h` parameter, pass through | Easy |
| 5. `meanflow_loss` (new) | **Implement loss with JVP for du/dt** | **Hard** |
| 6. `sample_tr` (lines 590-604) | Add data_proportion masking | Medium |
| 7. `training_step:448` | Switch from `rf_loss` to `meanflow_loss` | Easy |
| 8. `sample_actions:565` | Pass `h` to `dit_forward` | Easy |

---

## Key Considerations

### 1. **JVP (Forward-mode AD) vs Backward-mode AD**

**CRITICAL:** Mean Flow REQUIRES `torch.func.jvp`, not `torch.autograd.grad`:

- **JVP (Forward-mode)**: Computes derivatives in a single forward pass by propagating tangent vectors
- **Backward-mode AD**: Requires a backward pass through the computation graph
- **Efficiency**: JVP is more efficient for this use case (one output, few inputs)
- **Correctness**: The tangent vectors `(v, dtdt, drdt)` must be set correctly for proper derivative computation

**Why JVP?**
- We need `∂u/∂t` where `u` is high-dimensional (action space) and `t` is 1D
- Forward-mode AD has complexity O(input_dim), backward-mode has O(output_dim)
- For this case: forward-mode is much faster

**PyTorch Requirements:**
```python
# Required for JVP
import torch.func

# Use within autocast block
with torch.amp.autocast("cuda", enabled=False):
    u_pred, dudt = torch.func.jvp(u_func, primals, tangents)
```

### 2. **Data Proportion**

The `data_proportion` parameter (default 0.75) controls what fraction of samples use instantaneous velocity (r=t) vs average velocity (r<t). Start with 0.75 as in the paper.

### 3. **Guidance and Conditional Dropout**

The JAX implementation includes:
- CFG (Classifier-Free Guidance) via `guidance_fn`
- Conditional dropout via `cond_drop`

You may want to implement these for improved sample quality, especially for language-conditioned tasks.

### 4. **Adaptive Weighting**

The loss uses adaptive weighting to balance samples:
```python
adp_wt = (loss.detach() + norm_eps) ** norm_p
loss = loss / adp_wt
```

Parameters: `norm_eps=0.01`, `norm_p=1.0`

### 5. **Sampling Strategy**

The official implementation uses **1-step sampling** where `h = t = 1`:
- Start from noise `z_1 ~ N(0, 1)`
- Predict average velocity `u(z_1, t=1, h=1)`
- Generate data `z_0 = z_1 - u`

This is much faster than multi-step RF sampling!

### 6. **Validation**

You may want to keep both `rf_loss` and `meanflow_loss` initially to compare:
- Training stability
- Sample quality
- Convergence speed
- Inference speed (1-step vs multi-step)

---

## Testing Steps

1. Implement decoder and verify forward pass works
2. Test `sample_tr` produces correct distributions
3. **Verify `torch.func.jvp` works correctly** - this is the most critical step!
4. Train on small dataset and monitor loss curves
5. Compare sample quality with original RF implementation
6. Test 1-step sampling performance

---

## References and Important Notes

### Official Implementations

- **JAX/Flax**: [github.com/Gsunshine/meanflow](https://github.com/Gsunshine/meanflow/tree/main)
- **PyTorch**: [github.com/Gsunshine/py-meanflow](https://github.com/Gsunshine/py-meanflow)
  - Key file: [meanflow/models/meanflow.py](https://github.com/Gsunshine/py-meanflow/blob/main/meanflow/models/meanflow.py)

### Critical Corrections to Original Guide

**Original mistake**: The first version of this guide incorrectly suggested using `torch.autograd.grad` (backward-mode automatic differentiation).

**Correct approach**: Use `torch.func.jvp` (forward-mode automatic differentiation) as shown in the official PyTorch implementation.

**Why this matters**:
1. **Efficiency**: JVP is O(input_dim), backward AD is O(output_dim). For Mean Flow, input_dim << output_dim.
2. **Correctness**: The tangent vector formulation `(v, dtdt=1, drdt=0)` is specific to forward-mode AD.
3. **Single pass**: JVP computes both `u` and `du/dt` in one forward pass, not requiring backward pass.

### Key Implementation Details

From the official PyTorch code:
```python
# Network function
def u_func(z, t, r):
    h = t - r
    return self.net(z, (t.view(-1), h.view(-1)), aug_cond)

# Tangent vectors
dtdt = torch.ones_like(t)   # ∂t/∂t = 1
drdt = torch.zeros_like(r)  # ∂r/∂t = 0

# JVP computation
with torch.amp.autocast("cuda", enabled=False):
    u_pred, dudt = torch.func.jvp(u_func, (z, t, r), (v, dtdt, drdt))
    u_tgt = (v - (t - r) * dudt).detach()
    loss = (u_pred - u_tgt)**2
```

**DO NOT** use this pattern (incorrect):
```python
# ❌ WRONG - Don't use backward-mode AD
t_requires_grad = t.requires_grad_(True)
u = self.dit_forward(zt, t_requires_grad, h, cond)
du_dt = torch.autograd.grad(u, t_requires_grad, ...)  # Incorrect!
```

**DO** use this pattern (correct):
```python
# ✅ CORRECT - Use forward-mode JVP
def u_func(z, t, r):
    return self.dit_forward(z, t, t-r, cond)

u, dudt = torch.func.jvp(
    u_func,
    (z, t, r),
    (torch.zeros_like(z), torch.ones_like(t), torch.zeros_like(r))
)
```