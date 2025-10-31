import argparse
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class SensorForceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states_csv: str,
        sensor_csv: str,
        target_col: int = 4,
        x_mean=None, x_std=None,
        y_mean=None, y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
    ):
        states_time, states_val = [], []
        with open(states_csv, "r") as f:
            for row in csv.reader(f):
                vals = [float(x) for x in row]
                if len(vals) <= target_col:
                    continue
                if vals[target_col] != -1:
                    states_time.append(vals[0])
                    states_val.append(vals[target_col])
        states_time = np.asarray(states_time, dtype=np.float64)
        states_val = np.asarray(states_val, dtype=np.float64)

        sensor_time, sensor_data = [], []
        with open(sensor_csv, "r") as f:
            for row in csv.reader(f):
                vals = [float(x) for x in row]
                sensor_time.append(vals[0])
                sensor_data.append(vals[1:])
        sensor_time = np.asarray(sensor_time, dtype=np.float64)
        sensor_data = np.asarray(sensor_data, dtype=np.float64)

        matched_sens, matched_y = [], []
        for i in range(len(sensor_time)):
            t = sensor_time[i]
            idx = np.argmin(np.abs(states_time - t))
            matched_sens.append(sensor_data[i])
            matched_y.append(states_val[idx])

        self.X = np.asarray(matched_sens, dtype=np.float32)
        self.Y = np.asarray(matched_y, dtype=np.float32).reshape(-1, 1)

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        if self.normalize_x:
            if x_mean is None or x_std is None:
                x_mean = self.X.mean(axis=0)
                x_std = self.X.std(axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            self.x_mean = x_mean.astype(np.float32)
            self.x_std = x_std.astype(np.float32)
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

        if self.normalize_y:
            if y_mean is None or y_std is None:
                y_mean = self.Y.mean(axis=0)
                y_std = self.Y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros(1, dtype=np.float32)
            self.y_std = np.ones(1, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, device=y.device) + torch.tensor(self.y_mean, device=y.device)
        return y * self.y_std + self.y_mean

class SensorSpatialDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states_csv: str,
        sensor_csv: str,
        z_thresh: float = 145.1,
        x_mean=None, x_std=None,
        y_mean=None, y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
    ):
        states_time, states_xyz = [], []
        with open(states_csv, "r") as f:
            for row in csv.reader(f):
                row = [float(v) for v in row]
                states_time.append(row[0])
                states_xyz.append(row[1:4])
        states_time = np.asarray(states_time, dtype=np.float64)
        states_xyz = np.asarray(states_xyz, dtype=np.float64)

        sensor_time, sensor_data = [], []
        with open(sensor_csv, "r") as f:
            for row in csv.reader(f):
                row = [float(v) for v in row]
                sensor_time.append(row[0])
                sensor_data.append(row[1:])
        sensor_time = np.asarray(sensor_time, dtype=np.float64)
        sensor_data = np.asarray(sensor_data, dtype=np.float64)

        matched_xyz, matched_sens, matched_mask = [], [], []
        for i in range(len(sensor_time)):
            st = sensor_time[i]
            idx = np.argmin(np.abs(states_time - st))
            xyz = states_xyz[idx]
            matched_xyz.append(xyz)
            matched_sens.append(sensor_data[i])
            matched_mask.append(xyz[2] < z_thresh)

        matched_xyz = np.asarray(matched_xyz, dtype=np.float64)
        matched_sens = np.asarray(matched_sens, dtype=np.float64)
        matched_mask = np.asarray(matched_mask, dtype=bool)

        self.X = matched_sens[matched_mask].astype(np.float32)
        self.Y = matched_xyz[matched_mask].astype(np.float32)

        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        if self.normalize_x:
            if x_mean is None or x_std is None:
                x_mean = self.X.mean(axis=0)
                x_std = self.X.std(axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            self.x_mean = x_mean.astype(np.float32)
            self.x_std = x_std.astype(np.float32)
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

        if self.normalize_y:
            if y_mean is None or y_std is None:
                y_mean = self.Y.mean(axis=0)
                y_std = self.Y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros(3, dtype=np.float32)
            self.y_std = np.ones(3, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, device=y.device) + torch.tensor(self.y_mean, device=y.device)
        return y * self.y_std + self.y_mean


class NewFormatSpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for new format where each CSV contains:
    timestamp,position,x_pos,y_pos,fx,fy,fz,tx,ty,tz,mag0_x,mag0_y,mag0_z,...,mag4_z
    
    This adapter loads multiple position files and creates a localization dataset.
    """
    def __init__(
        self,
        data_dir: str,
        pattern: str = "position_*.csv",
        x_mean=None, x_std=None,
        y_mean=None, y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
        z_value: float = 0.0,  # Fixed z for 2D grid, or use from contact detection
    ):
        from pathlib import Path
        import glob
        
        data_path = Path(data_dir)
        csv_files = sorted(glob.glob(str(data_path / pattern)))
        
        if not csv_files:
            raise ValueError(f"No files matching {pattern} found in {data_dir}")
        
        all_sensors, all_positions = [], []
        
        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Extract position (x_pos, y_pos)
                        x_pos = float(row['x_pos'])
                        y_pos = float(row['y_pos'])
                        
                        # Extract magnetometer readings (columns 10-24, after tz)
                        mag_readings = [
                            float(row['mag0_x']), float(row['mag0_y']), float(row['mag0_z']),
                            float(row['mag1_x']), float(row['mag1_y']), float(row['mag1_z']),
                            float(row['mag2_x']), float(row['mag2_y']), float(row['mag2_z']),
                            float(row['mag3_x']), float(row['mag3_y']), float(row['mag3_z']),
                            float(row['mag4_x']), float(row['mag4_y']), float(row['mag4_z']),
                        ]
                        
                        # Skip if all magnetometer readings are zero (sensor not active)
                        if all(abs(m) < 1e-6 for m in mag_readings):
                            continue
                        
                        all_sensors.append(mag_readings)
                        all_positions.append([x_pos, y_pos, z_value])
                    except (KeyError, ValueError) as e:
                        # Skip malformed rows
                        continue
        
        if len(all_sensors) == 0:
            raise ValueError(f"No valid data found in {data_dir}")
        
        self.X = np.asarray(all_sensors, dtype=np.float32)
        self.Y = np.asarray(all_positions, dtype=np.float32)
        
        print(f"Loaded {len(self.X)} samples from {len(csv_files)} files in {data_dir}")
        
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        
        if self.normalize_x:
            if x_mean is None or x_std is None:
                x_mean = self.X.mean(axis=0)
                x_std = self.X.std(axis=0)
            x_std = np.where(x_std < 1e-8, 1.0, x_std)
            self.x_mean = x_mean.astype(np.float32)
            self.x_std = x_std.astype(np.float32)
            self.X = (self.X - self.x_mean) / self.x_std
        else:
            self.x_mean = np.zeros(self.X.shape[1], dtype=np.float32)
            self.x_std = np.ones(self.X.shape[1], dtype=np.float32)

        if self.normalize_y:
            if y_mean is None or y_std is None:
                y_mean = self.Y.mean(axis=0)
                y_std = self.Y.std(axis=0)
            y_std = np.where(y_std < 1e-8, 1.0, y_std)
            self.y_mean = y_mean.astype(np.float32)
            self.y_std = y_std.astype(np.float32)
            self.Y = (self.Y - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros(3, dtype=np.float32)
            self.y_std = np.ones(3, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return y * torch.tensor(self.y_std, device=y.device) + torch.tensor(self.y_mean, device=y.device)
        return y * self.y_std + self.y_mean

def fit(
    dataset_full,
    out_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
    test_split: float = 0.15,  # Add test split ratio
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset_full.X)
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    
    # Split into train/val/test: train gets remaining after val+test
    val_split = int((1.0 - test_split) * 0.2 * n)  # 20% of remaining becomes val
    train_split = int((1.0 - test_split) * 0.8 * n)  # 80% of remaining becomes train
    train_idx = idxs[:train_split]
    val_idx = idxs[train_split:train_split + val_split]
    test_idx = idxs[train_split + val_split:]

    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    x_mean = dataset_full.X[train_idx].mean(axis=0)
    x_std = dataset_full.X[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    y_mean = dataset_full.Y[train_idx].mean(axis=0)
    y_std = dataset_full.Y[train_idx].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    if out_dim == 1:
        # Force regression (DEPRECATED - should not reach here)
        dataset = SensorForceDataset(
            states_csv=dataset_full._states_csv,
            sensor_csv=dataset_full._sensor_csv,
            target_col=dataset_full._target_col,
            x_mean=x_mean, x_std=x_std,
            y_mean=y_mean, y_std=y_std,
            normalize_x=True, normalize_y=True,
        )
    else:
        # Localization (spatial)
        if hasattr(dataset_full, '_data_dir'):
            # NewFormatSpatialDataset
            dataset = NewFormatSpatialDataset(
                data_dir=dataset_full._data_dir,
                pattern=dataset_full._pattern,
                z_value=dataset_full._z_value,
                x_mean=x_mean, x_std=x_std,
                y_mean=y_mean, y_std=y_std,
                normalize_x=True, normalize_y=True,
            )
        else:
            # Legacy SensorSpatialDataset
            dataset = SensorSpatialDataset(
                states_csv=dataset_full._states_csv,
                sensor_csv=dataset_full._sensor_csv,
                z_thresh=dataset_full._z_thresh,
                x_mean=x_mean, x_std=x_std,
                y_mean=y_mean, y_std=y_std,
                normalize_x=True, normalize_y=True,
            )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx), batch_size=batch_size, shuffle=False
    )

    model = MLP(in_dim=dataset.X.shape[1], out_dim=out_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Store final metrics
    final_train_mse = None
    final_val_mse = None
    final_train_rmse = None
    final_val_rmse = None
    final_train_metrics = None
    final_val_metrics = None

    pbar = tqdm(range(1, epochs + 1), desc="Training", ncols=150)
    for e in pbar:
        model.train()
        train_loss_sum, train_count = 0.0, 0
        train_all_pred, train_all_true = [], []
        for Xb, Yb in train_loader:
            Xb = Xb.float().to(device)
            Yb = Yb.float().to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, Yb)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * Xb.size(0)
            train_count += Xb.size(0)
            train_all_pred.append(pred.detach().cpu())
            train_all_true.append(Yb.cpu())
        train_mse = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum, val_count = 0.0, 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for Xb, Yb in val_loader:
                Xb = Xb.float().to(device)
                Yb = Yb.float().to(device)
                pred = model(Xb)
                loss = criterion(pred, Yb)
                val_loss_sum += loss.item() * Xb.size(0)
                val_count += Xb.size(0)
                all_pred.append(pred.cpu())
                all_true.append(Yb.cpu())
        val_mse = val_loss_sum / max(1, val_count)

        pred = torch.cat(all_pred, dim=0)
        true = torch.cat(all_true, dim=0)

        pred_real = torch.from_numpy(dataset.unnormalize_y(pred.numpy()))
        true_real = torch.from_numpy(dataset.unnormalize_y(true.numpy()))
        d = pred_real - true_real

        if out_dim == 1:
            rmse = torch.sqrt(torch.mean(d[:, 0] ** 2)).item()
            pbar.set_postfix({
                'Train MSE': f'{train_mse:.4f}',
                'Val MSE': f'{val_mse:.4f}',
                'RMSE': f'{rmse:.3f}g'
            })
            final_train_mse = train_mse
            final_val_mse = val_mse
            final_val_rmse = rmse
        else:
            per_axis_rmse = torch.sqrt(torch.mean(d ** 2, dim=0))
            euclid_rmse = torch.sqrt(torch.mean(torch.sum(d ** 2, dim=1)))
            rx, ry, rz = (per_axis_rmse[0].item(), per_axis_rmse[1].item(), per_axis_rmse[2].item())
            pbar.set_postfix({
                'RMSE_x': f'{rx:.2f}mm',
                'RMSE_y': f'{ry:.2f}mm', 
                'RMSE_z': f'{rz:.2f}mm',
                'Net': f'{euclid_rmse:.2f}mm'
            })
            final_train_mse = train_mse
            final_val_mse = val_mse
            final_val_metrics = {
                'rmse_x': rx,
                'rmse_y': ry,
                'rmse_z': rz,
                'rmse_euclid': euclid_rmse.item()
            }
        
        # Calculate training RMSE for final epoch
        if e == epochs:
            train_pred_cat = torch.cat(train_all_pred, dim=0)
            train_true_cat = torch.cat(train_all_true, dim=0)
            train_pred_real = torch.from_numpy(dataset.unnormalize_y(train_pred_cat.numpy()))
            train_true_real = torch.from_numpy(dataset.unnormalize_y(train_true_cat.numpy()))
            train_d = train_pred_real - train_true_real
            
            if out_dim == 1:
                final_train_rmse = torch.sqrt(torch.mean(train_d[:, 0] ** 2)).item()
            else:
                train_per_axis_rmse = torch.sqrt(torch.mean(train_d ** 2, dim=0))
                train_euclid_rmse = torch.sqrt(torch.mean(torch.sum(train_d ** 2, dim=1)))
                final_train_metrics = {
                    'rmse_x': train_per_axis_rmse[0].item(),
                    'rmse_y': train_per_axis_rmse[1].item(),
                    'rmse_z': train_per_axis_rmse[2].item(),
                    'rmse_euclid': train_euclid_rmse.item()
                }

    # Evaluate on test set
    print("\n" + "="*60)
    print("Evaluating on TEST set...")
    print("="*60)
    model.eval()
    test_loss_sum, test_count = 0.0, 0
    test_all_pred, test_all_true = [], []
    with torch.no_grad():
        for Xb, Yb in test_loader:
            Xb = Xb.float().to(device)
            Yb = Yb.float().to(device)
            pred = model(Xb)
            loss = criterion(pred, Yb)
            test_loss_sum += loss.item() * Xb.size(0)
            test_count += Xb.size(0)
            test_all_pred.append(pred.cpu())
            test_all_true.append(Yb.cpu())
    
    test_mse = test_loss_sum / max(1, test_count)
    test_pred = torch.cat(test_all_pred, dim=0)
    test_true = torch.cat(test_all_true, dim=0)
    test_pred_real = torch.from_numpy(dataset.unnormalize_y(test_pred.numpy()))
    test_true_real = torch.from_numpy(dataset.unnormalize_y(test_true.numpy()))
    test_d = test_pred_real - test_true_real

    if out_dim == 1:
        test_rmse = torch.sqrt(torch.mean(test_d[:, 0] ** 2)).item()
    else:
        test_per_axis_rmse = torch.sqrt(torch.mean(test_d ** 2, dim=0))
        test_euclid_rmse = torch.sqrt(torch.mean(torch.sum(test_d ** 2, dim=1)))
        test_metrics = {
            'rmse_x': test_per_axis_rmse[0].item(),
            'rmse_y': test_per_axis_rmse[1].item(),
            'rmse_z': test_per_axis_rmse[2].item(),
            'rmse_euclid': test_euclid_rmse.item()
        }

    # Print comprehensive results summary
    print("\n" + "="*60)
    print("FINAL TRAINING RESULTS SUMMARY")
    print("="*60)
    
    if out_dim == 1:
        print(f"\nTraining Set:")
        print(f"  MSE:  {final_train_mse:.6f}")
        print(f"  RMSE: {final_train_rmse:.3f} g")
        print(f"\nValidation Set:")
        print(f"  MSE:  {final_val_mse:.6f}")
        print(f"  RMSE: {final_val_rmse:.3f} g")
        print(f"\nTest Set:")
        print(f"  MSE:  {test_mse:.6f}")
        print(f"  RMSE: {test_rmse:.3f} g")
    else:
        print(f"\nTraining Set:")
        print(f"  MSE:       {final_train_mse:.6f}")
        print(f"  RMSE (X):  {final_train_metrics['rmse_x']:.2f} mm")
        print(f"  RMSE (Y):  {final_train_metrics['rmse_y']:.2f} mm")
        print(f"  RMSE (Z):  {final_train_metrics['rmse_z']:.2f} mm")
        print(f"  RMSE (3D): {final_train_metrics['rmse_euclid']:.2f} mm")
        
        print(f"\nValidation Set:")
        print(f"  MSE:       {final_val_mse:.6f}")
        print(f"  RMSE (X):  {final_val_metrics['rmse_x']:.2f} mm")
        print(f"  RMSE (Y):  {final_val_metrics['rmse_y']:.2f} mm")
        print(f"  RMSE (Z):  {final_val_metrics['rmse_z']:.2f} mm")
        print(f"  RMSE (3D): {final_val_metrics['rmse_euclid']:.2f} mm")
        
        print(f"\nTest Set:")
        print(f"  MSE:       {test_mse:.6f}")
        print(f"  RMSE (X):  {test_metrics['rmse_x']:.2f} mm")
        print(f"  RMSE (Y):  {test_metrics['rmse_y']:.2f} mm")
        print(f"  RMSE (Z):  {test_metrics['rmse_z']:.2f} mm")
        print(f"  RMSE (3D): {test_metrics['rmse_euclid']:.2f} mm")
    
    print("="*60 + "\n")

    return model, (x_mean, x_std, y_mean, y_std)

def main():
    parser = argparse.ArgumentParser(description="eFlesh characterization training - LOCALIZATION ONLY")
    parser.add_argument("--mode", choices=["spatial", "newformat"], default="newformat",
                        help="Dataset format: 'spatial' (legacy) or 'newformat' (Data/local_sin_3*3/)")
    parser.add_argument("--folder", type=str, required=True, 
                        help="Path to dataset: for 'spatial', a probe folder; for 'newformat', directory with position_*.csv")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--z_thresh", type=float, default=145.1, help="Z threshold for legacy spatial mode")
    parser.add_argument("--z_value", type=float, default=0.0, help="Fixed Z value for newformat (2D grid)")
    parser.add_argument("--pattern", type=str, default="position_*.csv", help="File pattern for newformat mode")
    parser.add_argument("--seed", type=int, default=0)
    
    # Force regression modes are DISABLED by default (moved to unused/)
    parser.add_argument("--enable-force-regression", action="store_true", 
                        help="[DEPRECATED] Enable force regression modes (normal/shear). Data moved to unused/")
    parser.add_argument("--force-mode", choices=["normal", "shear"], 
                        help="[DEPRECATED] Force regression mode (requires --enable-force-regression)")
    parser.add_argument("--target_col", type=int, default=4, 
                        help="[DEPRECATED] Force column in states.csv")
    
    args = parser.parse_args()
    
    # Guard force regression modes
    if args.enable_force_regression:
        if not args.force_mode:
            print("ERROR: --force-mode required when --enable-force-regression is set")
            print("Force regression datasets have been moved to unused/characterization/datasets/")
            return
        print("WARNING: Force regression is deprecated. Datasets moved to unused/")
        print("Use legacy code from unused/experiments/ if you need this functionality.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # LOCALIZATION-ONLY MODES
    if args.mode == "newformat":
        # New format: Data/local_sin_3*3/ with position_*.csv files
        full = NewFormatSpatialDataset(
            data_dir=args.folder,
            pattern=args.pattern,
            z_value=args.z_value,
            normalize_x=False,
            normalize_y=False,
        )
        full._data_dir = args.folder
        full._pattern = args.pattern
        full._z_value = args.z_value
        out_dim = 3
        
    elif args.mode == "spatial":
        # Legacy format: characterization/datasets/spatial_resolution/
        states_csv = os.path.join(args.folder, "states.csv")
        sensor_csv = os.path.join(args.folder, "sensor_post_baselines.csv")
        
        if not os.path.exists(states_csv) or not os.path.exists(sensor_csv):
            print(f"ERROR: Legacy spatial mode requires states.csv and sensor_post_baselines.csv in {args.folder}")
            return
            
        full = SensorSpatialDataset(
            states_csv, sensor_csv, z_thresh=args.z_thresh,
            normalize_x=False, normalize_y=False,
        )
        full._states_csv = states_csv
        full._sensor_csv = sensor_csv
        full._z_thresh = args.z_thresh
        out_dim = 3

    print(f"\n{'='*60}")
    print(f"Starting LOCALIZATION training")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.folder}")
    print(f"Samples: {len(full.X)}")
    print(f"Input dim: {full.X.shape[1]}, Output dim: {out_dim}")
    print(f"{'='*60}\n")

    model, stats = fit(
        dataset_full=full,
        out_dim=out_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    # Save checkpoint
    artifacts_dir = os.path.join(args.folder, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(artifacts_dir, f"eflesh_localization_{args.mode}_mlp128.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "mode": args.mode,
            "out_dim": out_dim,
            "x_mean": stats[0],
            "x_std": stats[1],
            "y_mean": stats[2],
            "y_std": stats[3],
        },
        checkpoint_path,
    )
    print(f"\n{'='*60}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
