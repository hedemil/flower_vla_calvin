
# Mean Flow Decoder Integration Guide

This document summarizes the key locations and changes needed to integrate a custom mean flow decoder into the DiT architecture.

---

## 1. Decoder Definition
**Location:** `_setup_dit_components` (lines 376-377)

**Current:**
```python
self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim).to(self.device)
```
Currently, this is a simple linear projection from `dit_dim` to the action dimension.

**Action:**
- Replace `nn.Linear` with your mean flow decoder architecture.

---

## 2. Decoder Usage
**Location:** `decode_actions` (lines 780-796)

**Current:**
```python
def decode_actions(self, z: torch.Tensor, action_type: torch.Tensor, valid_dims: torch.Tensor) -> torch.Tensor:
	# ...
	for action_name, action_idx in self.action_space_index.action_spaces.items():
		mask = (action_type == action_idx)
		if mask.any():
			action_dim = self.action_space_index.get_action_dim(action_idx)
			pred = self.action_decoders[action_name]
			decoded = pred
			return decoded
```

**Action:**
- Update decoding logic if your decoder has a different signature.

---

## 3. Where Decoding Happens
**Location:** `dit_forward` (line 630)

**Current:**
```python
return self.decode_actions(cx, action_type, valid_dims)
```
Called at the end of the DiT forward pass after processing through all transformer blocks.

**Action:**
- If your decoder needs additional inputs (e.g., timestep `t`, conditioning), pass those through `dit_forward` to `decode_actions`.

---

## 4. Related Flow Methods to Consider

- **`rf_loss`** (lines 495-544): Computes rectified flow loss. The decoder output feeds into the loss computation via:
	```python
	vtheta = self.dit_forward(zt, t, cond)
	```
- **`sample_actions`** (lines 546-566): Uses Euler integration with `dit_forward` predictions.

---

## 5. Summary of Changes Needed

| Location                        | What to Modify                                                      |
|----------------------------------|---------------------------------------------------------------------|
| `_setup_dit_components:376-377`  | Replace `nn.Linear` with your mean flow decoder architecture        |
| `decode_actions:780-796`         | Update decoding logic if your decoder has a different signature     |
| `dit_forward:630` (if needed)    | Pass additional inputs (e.g., timestep, conditioning) if required   |

If your mean flow decoder requires additional inputs like the timestep or conditioning, you'll need to pass those through `dit_forward` to `decode_actions`.