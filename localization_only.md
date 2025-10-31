# eFlesh Localization-Only Migration Report

**Date:** October 28, 2025  
**Purpose:** Refactor eFlesh repository to support touch localization only, adapted for new data format (`Data/local_sin_3*3/`)

---

## Executive Summary

This migration successfully isolates the **touch localization** pipeline from force regression and slip detection experiments. The repository now defaults to localization-only training with support for a new single-CSV data format while preserving the original codebase in `unused/` for reference.

**Key Changes:**
- ✅ Force regression datasets moved to `unused/characterization/datasets/`
- ✅ Slip detection experiment moved to `unused/experiments/slip_detection/`
- ✅ New dataset adapter created for `Data/local_sin_3*3/` format
- ✅ Training script modified to disable force modes by default
- ✅ Localization pipeline is now the primary execution path

---

## 1. What's Reusable vs. Adapted

### ✅ Reusable As-Is

| Component | Location | Purpose | Notes |
|-----------|----------|---------|-------|
| **MLP Model** | `characterization/model.py` | Simple 3-layer neural network | Generic architecture, no changes needed |
| **Training Loop** | `characterization/train.py::fit()` | MSE loss + Adam optimizer | Modified to support new dataset type |
| **Metrics** | Training progress bars | Per-axis RMSE (x, y, z) + Euclidean distance | Appropriate for localization |
| **Visualization** | `visualizer/viz_eflesh.py` | Real-time magnetometer display | Useful for debugging sensors |
| **Hardware** | `arduino/5X_eflesh_stream/` | Firmware for 5 magnetometers | Required for data collection |
| **Dependencies** | `env.yml` | Conda environment | No changes required |

### ⚠️ Adapted for New Data Format

| Component | Status | Changes Made |
|-----------|--------|--------------|
| **Dataset Class** | ✅ **New adapter created** | `NewFormatSpatialDataset` reads single CSV files with columns: `timestamp`, `position`, `x_pos`, `y_pos`, `fx`, `fy`, `fz`, `tx`, `ty`, `tz`, `mag0_x`...`mag4_z` |
| **CLI Interface** | ✅ **Modified** | New `--mode newformat` flag, force modes guarded behind deprecated flags |
| **Training Entry** | ✅ **Simplified** | Defaults to localization, clear error messages for deprecated modes |

---

## 2. Files Moved to `unused/`

### Force Regression Datasets
**Destination:** `unused/characterization/datasets/`

```
unused/characterization/datasets/
├── normal_force/
│   ├── 20250401_014340_probe/  (states.csv, sensor.csv, sensor_post_baselines.csv)
│   ├── 20250401_015037_probe/
│   └── 20250401_015717_probe/
└── shear_force/
    ├── 20250416_200533_probe/  (includes .gif animations)
    └── 20250416_201911_probe/
```

**Reason:** These datasets contain force labels in `states.csv` (normal force in column 4, shear in other columns) and are used for regression tasks, not localization.

**Total files moved:** 19 files (9 CSVs for normal force, 10 files including gifs/pngs for shear force)

---

### Slip Detection Experiment
**Destination:** `unused/experiments/slip_detection/`

**Entire directory moved:**
```
unused/experiments/slip_detection/
├── train.py                    # LSTM-based binary classifier
├── model/lstm_model.py         # Time-series architecture
├── data/                       # 137 .pt sequence files + labels
├── checkpoints/                # Pre-trained slip models
├── robot-server/               # Robot control infrastructure
├── configs/                    # YAML configs for slip tasks
└── requirements.txt
```

**Reason:** Slip detection is a separate downstream application using LSTM models over time windows. It's a classification task (slip vs. no-slip), not spatial localization.

**Total files moved:** ~200+ files

---

### Other Cleanup
- `visuoskin/` (empty directory) → **removed**

---

## 3. API Changes

### Dataset Signatures

#### **NEW: `NewFormatSpatialDataset`** (Recommended)
```python
dataset = NewFormatSpatialDataset(
    data_dir="Data/local_sin_3*3/",      # Directory with position_*.csv
    pattern="position_*.csv",             # File glob pattern
    z_value=0.0,                          # Fixed Z for 2D grid
    x_mean=None, x_std=None,              # Optional normalization stats
    y_mean=None, y_std=None,
    normalize_x=True,
    normalize_y=True,
)
```

**Input CSV Format:**
```csv
timestamp,position,x_pos,y_pos,fx,fy,fz,tx,ty,tz,mag0_x,mag0_y,mag0_z,...,mag4_z
1761629728.99,2,2,1,-0.3,-0.15,-0.48,0.0008,0.002,-0.0009,-1.44,3.84,-1.55,...
```

- **Extracts:** `x_pos`, `y_pos` (positions), `mag0_x` through `mag4_z` (15 magnetometer readings)
- **Ignores:** `fx`, `fy`, `fz`, `tx`, `ty`, `tz` (force/torque data)
- **Filters:** Rows with all-zero magnetometer readings

---

#### **Legacy: `SensorSpatialDataset`** (Still Supported)
```python
dataset = SensorSpatialDataset(
    states_csv="characterization/datasets/spatial_resolution/.../states.csv",
    sensor_csv=".../sensor_post_baselines.csv",
    z_thresh=145.1,                       # Z threshold for contact detection
    x_mean=None, x_std=None,
    y_mean=None, y_std=None,
    normalize_x=True,
    normalize_y=True,
)
```

**Input Format (two separate CSVs):**
- `states.csv`: `timestamp, x, y, z`
- `sensor_post_baselines.csv`: `timestamp, mag0_x, ..., mag4_z`

---

### CLI Interface Changes

#### **New Default Mode: `--mode newformat`**
```bash
# Train on your new data (default mode)
python characterization/train.py \
    --folder Data/local_sin_3*3/ \
    --epochs 500 \
    --batch_size 64

# Explicit mode specification
python characterization/train.py \
    --mode newformat \
    --folder Data/local_sin_3*3/ \
    --pattern "position_*.csv" \
    --z_value 0.0
```

#### **Legacy Mode: `--mode spatial`**
```bash
# Use original format (reference datasets)
python characterization/train.py \
    --mode spatial \
    --folder characterization/datasets/spatial_resolution/20250316_155729_probe/ \
    --z_thresh 145.1
```

#### **Deprecated: Force Regression Modes** ❌
```bash
# These now return an error message
python characterization/train.py \
    --enable-force-regression \
    --force-mode normal \
    --folder unused/characterization/datasets/normal_force/...

# Error message:
# "WARNING: Force regression is deprecated. Datasets moved to unused/"
# "Use legacy code from unused/experiments/ if you need this functionality."
```

---

### Config Keys

| Key | Old Default | New Default | Notes |
|-----|-------------|-------------|-------|
| `--mode` | (required) | `newformat` | Now has a default |
| `--folder` | (required) | (still required) | Path semantics changed |
| `--z_value` | N/A | `0.0` | For 2D grids in new format |
| `--pattern` | N/A | `position_*.csv` | File matching pattern |
| `--enable-force-regression` | N/A | `False` | Guard for deprecated modes |

---

## 4. How to Run Localization Training

### Quick Start (Your Data)

```bash
# 1. Activate environment
conda activate eflesh

# 2. Train localization model on Data/local_sin_3*3/
python characterization/train.py --folder Data/local_sin_3*3/ --epochs 200

# Expected output:
# Loaded 1998 samples from 9 files in Data/local_sin_3*3/
# Starting LOCALIZATION training
# Mode: newformat
# Dataset: Data/local_sin_3*3/
# Samples: 1998
# Input dim: 15, Output dim: 3
# ... training progress bars with RMSE_x, RMSE_y, RMSE_z, Net ...
# Model saved to: Data/local_sin_3*3/artifacts/eflesh_localization_newformat_mlp128.pt
```

### Training Parameters

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--epochs` | 1000 | Training iterations | Start with 200-500 for testing |
| `--batch_size` | 64 | Samples per batch | Reduce if out of memory |
| `--lr` | 1e-3 | Learning rate | Good default for Adam |
| `--seed` | 0 | Random seed | For reproducibility |
| `--z_value` | 0.0 | Fixed Z coordinate | Adjust if 2D grid has known height |

---

### Evaluation Metrics

During training, the script reports:
- **RMSE_x, RMSE_y, RMSE_z:** Per-axis localization error (in mm, assuming positions in mm)
- **Net:** Euclidean distance error √(Δx² + Δy² + Δz²)

**Example output:**
```
Epoch 100/200: RMSE_x: 2.34mm, RMSE_y: 1.87mm, RMSE_z: 0.00mm, Net: 3.01mm
```

---

### Checkpoint Format

```python
checkpoint = {
    "state_dict": model.state_dict(),     # PyTorch model weights
    "mode": "newformat",                  # Dataset format used
    "out_dim": 3,                         # Output dimension (x, y, z)
    "x_mean": np.array([...]),            # Magnetometer normalization
    "x_std": np.array([...]),
    "y_mean": np.array([...]),            # Position normalization
    "y_std": np.array([...]),
}
```

**Load for inference:**
```python
import torch
from characterization.model import MLP

checkpoint = torch.load("Data/local_sin_3*3/artifacts/eflesh_localization_newformat_mlp128.pt")
model = MLP(in_dim=15, out_dim=3, hidden=128)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Normalize input, predict, denormalize output
sensor_data = ...  # shape: (15,)
x_norm = (sensor_data - checkpoint["x_mean"]) / checkpoint["x_std"]
y_norm = model(torch.tensor(x_norm, dtype=torch.float32))
y_pred = y_norm.numpy() * checkpoint["y_std"] + checkpoint["y_mean"]  # (x, y, z)
```

---

## 5. Data Format Details

### Your Data (`Data/local_sin_3*3/`)

**Structure:**
```
Data/local_sin_3*3/
├── position_1_20251028_003401.csv
├── position_2_20251028_003525.csv
├── ...
└── position_9_20251028_004642.csv
```

**Schema:**
```
Column Index | Name      | Type   | Purpose                | Usage in Localization
-------------|-----------|--------|------------------------|---------------------
0            | timestamp | float  | Time in seconds        | Ignored (single-frame)
1            | position  | int    | Grid position ID       | Metadata (not used in model)
2            | x_pos     | float  | X coordinate           | ✅ Target output
3            | y_pos     | float  | Y coordinate           | ✅ Target output
4-9          | fx,fy,fz,tx,ty,tz | float | Forces/torques | ❌ Ignored (force regression only)
10-24        | mag0_x...mag4_z | float | Magnetometer xyz | ✅ Model input (15 features)
```

**Key Insights:**
- Each CSV corresponds to a **fixed position** in a grid (e.g., position 1 = top-left).
- Multiple rows per position represent **temporal samples** at that location.
- The adapter treats each row as an **independent training sample** (x_pos, y_pos → mag readings).

**Assumptions for 2D Grid:**
- `z_value=0.0` is used since your data doesn't have a `z_pos` column.
- If your grid has known heights, adjust `--z_value` accordingly.

---

### Original Format (Reference)

Located in `characterization/datasets/spatial_resolution/`:

**Two-file structure:**
```
20250316_155729_probe/
├── states.csv                    # timestamp, x, y, z
└── sensor_post_baselines.csv     # timestamp, mag0_x, ..., mag4_z
```

**Time-alignment:** The `SensorSpatialDataset` matches sensor readings to positions by finding the nearest timestamp.

---

## 6. What to Edit Next (If Sensor Layout Changes)

### 1. **Number of Magnetometers**
**Current:** 5 magnetometers (15 features: 5 × 3 axes)

**If you change to N magnetometers:**
1. Update `NewFormatSpatialDataset.__init__()`:
   ```python
   mag_readings = [
       float(row['mag0_x']), float(row['mag0_y']), float(row['mag0_z']),
       # ... add/remove magnetometers here
   ]
   ```
2. Update CSV column names if needed.
3. Model input dimension will auto-adjust (it's inferred from `dataset.X.shape[1]`).

---

### 2. **Magnetometer Layout / Orientation**
**Current:** Magnetometers named `mag0` through `mag4`.

**If layout changes:**
- Update column names in your CSV export script.
- Update the `mag_readings` extraction in `NewFormatSpatialDataset`.
- Consider adding a **calibration step** if relative positions matter (e.g., baseline subtraction, axis rotation).

---

### 3. **2D vs. 3D Localization**
**Current:** 2D grid (x, y) with fixed z.

**To switch to 3D:**
1. Add `z_pos` column to your CSV.
2. Modify `NewFormatSpatialDataset`:
   ```python
   z_pos = float(row.get('z_pos', z_value))  # Use z_pos if available
   all_positions.append([x_pos, y_pos, z_pos])
   ```
3. Remove `--z_value` CLI argument or make it optional.

---

### 4. **Grid Geometry**
**Current:** 3×3 grid (9 positions).

**For different grids:**
- Just provide more/fewer `position_*.csv` files.
- The adapter automatically loads all matching files.
- Grid shape doesn't need to be specified (the model learns continuous x, y mapping).

---

### 5. **Force Integration (Future)**
**If you want to re-enable force prediction:**
1. Restore force datasets from `unused/characterization/datasets/`.
2. Create a **multi-task model** with two heads:
   - Head 1: Localization (x, y, z) — MSE loss
   - Head 2: Force (fx, fy, fz) — MSE loss
3. Modify loss function to combine both tasks:
   ```python
   loss = loss_localization + λ * loss_force
   ```
4. Update `NewFormatSpatialDataset` to return both targets:
   ```python
   def __getitem__(self, idx):
       return self.X[idx], (self.Y_pos[idx], self.Y_force[idx])
   ```

---

## 7. Repository Structure (After Migration)

```
eFlesh/
├── characterization/              ✅ ACTIVE (localization only)
│   ├── train.py                  (modified: newformat mode, force modes disabled)
│   ├── model.py                  (unchanged: generic MLP)
│   └── datasets/
│       └── spatial_resolution/   (reference datasets, legacy format)
│
├── Data/                          ✅ YOUR DATA (new format)
│   └── local_sin_3*3/            (9 position CSVs)
│
├── visualizer/                    ✅ USEFUL (debugging)
│   ├── viz_eflesh.py             (real-time sensor visualization)
│   └── viz_fingertip.py
│
├── arduino/                       ✅ HARDWARE (firmware)
│   └── 5X_eflesh_stream/
│
├── microstructure/                ⚙️ DESIGN TOOLS (separate from ML)
│   ├── matopt/                   (optimization framework)
│   └── microstructure_inflators/ (mesh generation)
│
├── unused/                        📦 ARCHIVED (reference only)
│   ├── characterization/datasets/
│   │   ├── normal_force/         (force regression data)
│   │   └── shear_force/
│   └── experiments/
│       └── slip_detection/       (entire slip detection project)
│
├── env.yml                        ✅ DEPENDENCIES
├── README.md                      ✅ UPDATED (quickstart added)
└── localization_only.md          📄 THIS DOCUMENT
```

---

## 8. Troubleshooting

### Issue: "No files matching pattern found"
**Solution:** Check that:
1. `--folder` points to the correct directory.
2. CSV files match `--pattern` (default: `position_*.csv`).
3. Files are not hidden or in subdirectories.

---

### Issue: "No valid data found"
**Cause:** All magnetometer readings are zero (sensor not active).

**Solution:**
1. Verify sensor connection using `visualizer/viz_eflesh.py`.
2. Check CSV files manually for non-zero magnetometer columns.
3. Ensure baseline subtraction was applied correctly during data collection.

---

### Issue: "KeyError: 'mag0_x'"
**Cause:** CSV column names don't match expected format.

**Solution:**
1. Open a CSV file and verify header row matches:
   ```
   timestamp,position,x_pos,y_pos,fx,fy,fz,tx,ty,tz,mag0_x,mag0_y,mag0_z,...
   ```
2. If names differ, modify `NewFormatSpatialDataset` to use your column names.

---

### Issue: High localization error (RMSE > 10mm)
**Debugging steps:**
1. **Data quality:** Plot `x_pos` vs. magnetometer readings to check correlation.
2. **Overfitting:** Check if train/val RMSE diverge (reduce model size or add regularization).
3. **Normalization:** Verify normalization stats are computed correctly.
4. **Calibration:** Ensure magnetometers were baseline-subtracted during collection.

---

## 9. Testing Checklist

- [x] Force regression datasets moved to `unused/`
- [x] Slip detection moved to `unused/`
- [x] `NewFormatSpatialDataset` loads `Data/local_sin_3*3/` correctly
- [x] CLI defaults to `--mode newformat`
- [x] Force modes return deprecation errors
- [x] Legacy `--mode spatial` still works on reference data
- [ ] Training completes successfully on your data (see next section)
- [ ] Checkpoint loads and runs inference
- [ ] No linter errors in `train.py`

---

## 10. Next Steps

### Immediate
1. **Test training:** Run `python characterization/train.py --folder Data/local_sin_3*3/ --epochs 50` to verify.
2. **Check metrics:** Ensure RMSE values are reasonable for your grid spacing.
3. **Validate checkpoint:** Load and test inference on held-out data.

### Short-Term
1. **Hyperparameter tuning:** Experiment with learning rate, hidden size, epochs.
2. **Data augmentation:** Add noise/jitter to magnetometer readings for robustness.
3. **Cross-validation:** Split data by position (e.g., train on 7 positions, test on 2).

### Long-Term
1. **Real-time inference:** Integrate trained model with Arduino sensor stream.
2. **Visualization:** Plot predicted vs. true positions to assess spatial coverage.
3. **Multi-task learning:** Add force prediction if needed (see Section 6.5).

---

## 11. Maintenance Notes

**For future contributors:**
- **Localization code:** `characterization/train.py` + `model.py`
- **Force/slip code:** See `unused/` for historical reference
- **Data format changes:** Update `NewFormatSpatialDataset` in `train.py`
- **Sensor changes:** Update magnetometer extraction in dataset `__init__` method

**Git History:**
- Force datasets moved with `git mv` to preserve history.
- Check `git log --follow unused/characterization/datasets/normal_force/` to see original commits.

---

## Summary

**What Changed:**
- Repository now focuses on **touch localization only**.
- New data format (`Data/local_sin_3*3/`) is the primary target.
- Force regression and slip detection moved to `unused/` but preserved for reference.

**What Stayed:**
- Original model architecture and training loop.
- Legacy dataset support for spatial resolution.
- Hardware and visualization tools.

**How to Proceed:**
1. Run training on your data: `python characterization/train.py --folder Data/local_sin_3*3/`
2. Check localization metrics (RMSE should be < 5mm for tight grids).
3. Iterate on hyperparameters or dataset size if needed.

**Questions?**
- Review CSV format in Section 5.
- Check troubleshooting in Section 8.
- Examine `unused/` for legacy force/slip experiments.

---

**End of Migration Report**


