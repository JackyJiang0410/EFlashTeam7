# Testing Guide for Localization Training

## Prerequisites

Before running the training, ensure your Python environment is set up:

### Option 1: Using Conda (Recommended)

```bash
# Create environment from env.yml
conda env create -f env.yml

# Activate environment
conda activate eflesh

# Verify installation
python -c "import torch; import numpy; print('Environment ready!')"
```

### Option 2: Using pip

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy tqdm matplotlib
```

---

## Quick Test (10 epochs)

Run a quick test to verify the pipeline works:

```bash
cd /Users/haojiang/Desktop/ECE382N/Project/eFlesh

python characterization/train.py \
    --folder Data/local_sin_3*3/ \
    --epochs 10 \
    --batch_size 32
```

**Expected behavior:**
1. Loads CSV files from `Data/local_sin_3*3/`
2. Prints dataset statistics (number of samples, input/output dimensions)
3. Shows training progress with RMSE metrics
4. Saves checkpoint to `Data/local_sin_3*3/artifacts/`

**Expected output:**
```
Loaded XXXX samples from 9 files in Data/local_sin_3*3/

============================================================
Starting LOCALIZATION training
Mode: newformat
Dataset: Data/local_sin_3*3/
Samples: XXXX
Input dim: 15, Output dim: 3
============================================================

Training: 100%|████████████| 10/10 [00:XX<00:00, X.XXs/it, RMSE_x=X.XXmm, RMSE_y=X.XXmm, RMSE_z=0.00mm, Net=X.XXmm]

============================================================
Model saved to: Data/local_sin_3*3/artifacts/eflesh_localization_newformat_mlp128.pt
============================================================
```

---

## Full Training

For production training, use more epochs:

```bash
python characterization/train.py \
    --folder Data/local_sin_3*3/ \
    --epochs 500 \
    --batch_size 64 \
    --lr 1e-3
```

---

## Verify Checkpoint

After training, verify the checkpoint was created:

```bash
ls -lh Data/local_sin_3*3/artifacts/

# Should show:
# eflesh_localization_newformat_mlp128.pt
```

---

## Load and Test Inference

```python
import torch
import numpy as np
from characterization.model import MLP

# Load checkpoint
checkpoint = torch.load("Data/local_sin_3*3/artifacts/eflesh_localization_newformat_mlp128.pt")

print(f"Mode: {checkpoint['mode']}")
print(f"Output dim: {checkpoint['out_dim']}")
print(f"Normalization stats available: x_mean, x_std, y_mean, y_std")

# Create model
model = MLP(in_dim=15, out_dim=3, hidden=128)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Test inference with dummy data
dummy_sensor = np.random.randn(15).astype(np.float32)
x_norm = (dummy_sensor - checkpoint["x_mean"]) / checkpoint["x_std"]
x_tensor = torch.tensor(x_norm, dtype=torch.float32)

with torch.no_grad():
    y_norm = model(x_tensor)
    
position = y_norm.numpy() * checkpoint["y_std"] + checkpoint["y_mean"]
print(f"Predicted position: x={position[0]:.2f}, y={position[1]:.2f}, z={position[2]:.2f}")
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Activate your conda/virtual environment first.

### Issue: "No files matching position_*.csv found"
**Solution:** Check that CSV files are in the correct directory:
```bash
ls Data/local_sin_3*3/
# Should show: position_1_*.csv, position_2_*.csv, ..., position_9_*.csv
```

### Issue: "KeyError: 'mag0_x'"
**Solution:** Verify CSV header matches expected format:
```bash
head -1 Data/local_sin_3*3/position_1_*.csv
# Should show: timestamp,position,x_pos,y_pos,fx,fy,fz,tx,ty,tz,mag0_x,mag0_y,mag0_z,...
```

### Issue: High RMSE (> 50mm)
**Possible causes:**
1. Data quality issues (check for sensor calibration)
2. Wrong units (ensure positions are in mm, not meters)
3. Need more training epochs
4. Sensor placement doesn't provide enough spatial resolution

---

## Code Verification Checklist

- [x] No syntax errors in `train.py` (verified with linter)
- [x] `NewFormatSpatialDataset` class created
- [x] CLI updated with `--mode newformat` default
- [x] Force modes disabled/deprecated
- [ ] Environment set up (conda/pip)
- [ ] Training completes without errors
- [ ] Checkpoint file created
- [ ] Inference code runs successfully

---

## Next Steps After Testing

1. **Hyperparameter tuning:** Adjust epochs, learning rate, batch size
2. **Cross-validation:** Split data spatially (train on subset of positions, test on others)
3. **Visualization:** Plot predicted vs. actual positions
4. **Real-time inference:** Integrate with Arduino sensor stream

---

## Contact

If you encounter issues not covered here, check:
- `localization_only.md` for detailed migration documentation
- `README.md` for general project information
- Original eFlesh paper for sensor design details


