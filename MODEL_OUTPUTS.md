# Model Outputs Summary

This document describes the output format and meaning for each model in the eFlesh characterization system.

## 1. Single-Touch Localization Model

**Script:** `characterization/train.py --mode single_touch`

**Output Dimension:** 3

**Output Format:** `[x, y, abs(fz)]`

| Index | Output | Unit | Description |
|-------|--------|------|-------------|
| 0 | `x` | grid | X-coordinate of contact position on the 2D grid |
| 1 | `y` | grid | Y-coordinate of contact position on the 2D grid |
| 2 | `abs(fz)` | N (Newtons) | Absolute value of normal force (force in z-direction) |

**Notes:**
- `x` and `y` are grid coordinates (typically 1-3 for a 3×3 grid)
- `abs(fz)` is the absolute value of the force, always non-negative
- The model predicts both position and force simultaneously
- After denormalization, outputs are in original units (grid for x/y, N for force)

**Example Output:**
```python
[2.1, 1.8, 3.45]  # x=2.1 grid, y=1.8 grid, force=3.45 N
```

---

## 2. Multi-Touch Localization Model

**Script:** `characterization/train.py --mode multi_touch`

**Output Dimension:** Variable (depends on number of contact points)

**Output Format:** `[pos1_x, pos1_y, pos2_x, pos2_y, pos3_x, pos3_y, ...]`

| Index | Output | Unit | Description |
|-------|--------|------|-------------|
| 0 | `pos1_x` | grid | X-coordinate of 1st contact point |
| 1 | `pos1_y` | grid | Y-coordinate of 1st contact point |
| 2 | `pos2_x` | grid | X-coordinate of 2nd contact point |
| 3 | `pos2_y` | grid | Y-coordinate of 2nd contact point |
| ... | ... | ... | Additional contact points (if any) |

**Notes:**
- Output is flattened: pairs of (x, y) coordinates for each contact point
- Number of contact points depends on the data (typically 2 touches = 4 outputs)
- Each contact point has 2 coordinates (x, y) - no force prediction for multi-touch
- Coordinates are in grid units
- The model can handle variable number of touches (limited by `max_touches` parameter)

**Example Output (2 touches):**
```python
[1.2, 2.3, 2.8, 1.5]  # Touch 1: (1.2, 2.3), Touch 2: (2.8, 1.5)
```

**Example Output (3 touches):**
```python
[1.2, 2.3, 2.8, 1.5, 1.9, 2.7]  # Touch 1: (1.2, 2.3), Touch 2: (2.8, 1.5), Touch 3: (1.9, 2.7)
```

---

## 3. Touch Classification Model

**Script:** `characterization/touch_classifier.py`

**Output Dimension:** 1

**Output Format:** Single logit value (raw output before sigmoid)

| Output | Type | Range | Description |
|--------|------|-------|-------------|
| `logit` | float | (-∞, +∞) | Raw logit score from the model |

**Interpretation:**
- **After sigmoid:** `sigmoid(logit)` gives probability of multi-touch
  - `sigmoid(logit) < 0.5` → Single-touch (class 0)
  - `sigmoid(logit) >= 0.5` → Multi-touch (class 1)
  
- **Direct logit:**
  - `logit < 0` → Single-touch (class 0)
  - `logit >= 0` → Multi-touch (class 1)

**Notes:**
- The model outputs a single scalar value (logit)
- To get probability: `probability = 1 / (1 + exp(-logit))` or use `torch.sigmoid(logit)`
- To get class prediction: `class = 1 if sigmoid(logit) >= 0.5 else 0`
- Higher positive values = higher confidence for multi-touch
- Lower negative values = higher confidence for single-touch

**Example Outputs:**
```python
logit = -2.5   → sigmoid(-2.5) = 0.076  → Single-touch (class 0)
logit = 0.0    → sigmoid(0.0) = 0.5     → Multi-touch (class 1, threshold)
logit = 3.2    → sigmoid(3.2) = 0.961   → Multi-touch (class 1, high confidence)
```

---

## Summary Table

| Model | Output Dim | Output Type | Units | Interpretation |
|-------|-----------|------------|-------|----------------|
| **Single-touch localization** | 3 | `[x, y, abs(fz)]` | grid, grid, N | Contact position (x, y) and force magnitude |
| **Multi-touch localization** | 2N (N = # touches) | `[pos1_x, pos1_y, pos2_x, pos2_y, ...]` | grid | Flattened (x, y) coordinates for each contact point |
| **Touch classifier** | 1 | `logit` | unitless | Logit score: < 0 = single-touch, >= 0 = multi-touch |

---

## Input for All Models

All models share the same input format:

**Input Dimension:** 15

**Input Format:** `[mag0_x, mag0_y, mag0_z, mag1_x, mag1_y, mag1_z, ..., mag4_z]`

- 15 magnetometer readings from 5 sensors (3 axes each: x, y, z)
- Units: Magnetic field strength (after baseline subtraction)
- Normalized using z-score normalization (mean=0, std=1) during training

---

## Post-Processing

### Denormalization

All model outputs (except classifier logit) need to be denormalized using statistics saved in the checkpoint:

```python
# For localization models
denormalized = output * y_std + y_mean

# For single-touch: y_mean and y_std are [3] arrays
# For multi-touch: y_mean and y_std are [2N] arrays
```

### Classification

For the classifier, apply sigmoid to get probability:

```python
probability = torch.sigmoid(logit)  # or 1 / (1 + exp(-logit))
predicted_class = 1 if probability >= 0.5 else 0
```


