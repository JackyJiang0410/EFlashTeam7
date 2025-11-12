import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .datasets import (
    MultiTouchSpatialDataset,
    SingleTouchSpatialDataset,
    SensorSpatialDataset,
)
from .models import MLP


def _prepare_dataset(
    dataset_full,
    dataset_type: str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
):
    if dataset_type == "multi_touch":
        return MultiTouchSpatialDataset(
            data_dir=dataset_full._data_dir,
            pattern=dataset_full._pattern,
            max_touches=dataset_full._max_touches,
            drop_zero_rows=dataset_full._drop_zero_rows,
            min_force_newton=dataset_full._min_force_newton,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            normalize_x=True,
            normalize_y=True,
        )

    if dataset_type == "single_touch":
        return SingleTouchSpatialDataset(
            data_dir=dataset_full._data_dir,
            pattern=dataset_full._pattern,
            z_value=dataset_full._z_value,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            normalize_x=True,
            normalize_y=True,
        )

    if dataset_type in (None, "spatial"):
        return SensorSpatialDataset(
            states_csv=dataset_full._states_csv,
            sensor_csv=dataset_full._sensor_csv,
            z_thresh=dataset_full._z_thresh,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            normalize_x=True,
            normalize_y=True,
        )

    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def _split_indices(n: int, test_split: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idxs = np.arange(n)
    np.random.shuffle(idxs)

    val_split = int((1.0 - test_split) * 0.2 * n)
    train_split = int((1.0 - test_split) * 0.8 * n)

    train_idx = idxs[:train_split]
    val_idx = idxs[train_split : train_split + val_split]
    test_idx = idxs[train_split + val_split :]

    return train_idx, val_idx, test_idx


def fit(
    dataset_full,
    out_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 0,
    test_split: float = 0.15,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(dataset_full.X)
    train_idx, val_idx, test_idx = _split_indices(n, test_split)
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    x_mean = dataset_full.X[train_idx].mean(axis=0)
    x_std = dataset_full.X[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    y_mean = dataset_full.Y[train_idx].mean(axis=0)
    y_std = dataset_full.Y[train_idx].std(axis=0)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    dataset_type = getattr(dataset_full, "_dataset_type", None)
    dataset = _prepare_dataset(dataset_full, dataset_type, x_mean, x_std, y_mean, y_std)

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

    final_train_mse = final_val_mse = None
    final_train_rmse = final_val_rmse = None
    final_train_metrics = final_val_metrics = None

    is_multi_touch = dataset_type == "multi_touch"
    touch_pairs: List[Tuple[str, str]] = dataset.touch_pairs if is_multi_touch else []

    pbar = tqdm(range(1, epochs + 1), desc="Training", ncols=150)

    for epoch in pbar:
        model.train()
        train_loss_sum = train_count = 0.0
        train_all_pred, train_all_true = [], []

        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)

            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
            train_all_pred.append(pred.detach().cpu())
            train_all_true.append(yb.cpu())

        train_mse = train_loss_sum / max(1, train_count)

        model.eval()
        val_loss_sum = val_count = 0.0
        all_pred, all_true = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                yb = yb.float().to(device)
                pred = model(xb)

                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)

                all_pred.append(pred.cpu())
                all_true.append(yb.cpu())

        val_mse = val_loss_sum / max(1, val_count)
        pred = torch.cat(all_pred, dim=0)
        true = torch.cat(all_true, dim=0)

        pred_real = torch.from_numpy(dataset.unnormalize_y(pred.numpy()))
        true_real = torch.from_numpy(dataset.unnormalize_y(true.numpy()))
        diff = pred_real - true_real

        if out_dim == 1:
            rmse = torch.sqrt(torch.mean(diff[:, 0] ** 2)).item()
            pbar.set_postfix(
                {"Train MSE": f"{train_mse:.4f}", "Val MSE": f"{val_mse:.4f}", "RMSE": f"{rmse:.3f}g"}
            )
            final_train_mse = train_mse
            final_val_mse = val_mse
            final_val_rmse = rmse

        elif is_multi_touch:
            diff_mt = diff.view(diff.size(0), len(touch_pairs), 2)
            per_coord_rmse: Dict[str, float] = {}
            per_coord_mse: Dict[str, float] = {}

            for idx, (x_key, y_key) in enumerate(touch_pairs):
                dx = diff_mt[:, idx, 0]
                dy = diff_mt[:, idx, 1]
                per_coord_rmse[x_key] = torch.sqrt(torch.mean(dx**2)).item()
                per_coord_rmse[y_key] = torch.sqrt(torch.mean(dy**2)).item()
                per_coord_mse[x_key] = torch.mean(dx**2).item()
                per_coord_mse[y_key] = torch.mean(dy**2).item()

            mse_all = torch.mean(torch.sum(diff_mt**2, dim=2)).item()
            rmse_all = math.sqrt(mse_all)
            postfix = {f"RMSE_{coord}": f"{per_coord_rmse[coord]:.2f} grid" for coord in per_coord_rmse}
            postfix["RMSE_all"] = f"{rmse_all:.2f} grid"
            pbar.set_postfix(postfix)

            final_train_mse = train_mse
            final_val_mse = val_mse
            final_val_metrics = {
                "per_coord_rmse": per_coord_rmse,
                "per_coord_mse": per_coord_mse,
                "rmse_all": rmse_all,
                "mse_all": mse_all,
            }
        else:
            per_axis_rmse = torch.sqrt(torch.mean(diff**2, dim=0))
            euclid_rmse = torch.sqrt(torch.mean(torch.sum(diff**2, dim=1)))
            rx, ry, rz = (
                per_axis_rmse[0].item(),
                per_axis_rmse[1].item(),
                per_axis_rmse[2].item(),
            )
            pbar.set_postfix(
                {
                    "RMSE_x": f"{rx:.2f} grid",
                    "RMSE_y": f"{ry:.2f} grid",
                    "RMSE_z": f"{rz:.2f} grid",
                    "Net": f"{euclid_rmse:.2f} grid",
                }
            )
            final_train_mse = train_mse
            final_val_mse = val_mse
            final_val_metrics = {
                "rmse_x": rx,
                "rmse_y": ry,
                "rmse_z": rz,
                "rmse_euclid": euclid_rmse.item(),
            }

        if epoch == epochs:
            train_pred_cat = torch.cat(train_all_pred, dim=0)
            train_true_cat = torch.cat(train_all_true, dim=0)
            train_pred_real = torch.from_numpy(dataset.unnormalize_y(train_pred_cat.numpy()))
            train_true_real = torch.from_numpy(dataset.unnormalize_y(train_true_cat.numpy()))
            train_diff = train_pred_real - train_true_real

            if out_dim == 1:
                final_train_rmse = torch.sqrt(torch.mean(train_diff[:, 0] ** 2)).item()
            elif is_multi_touch:
                diff_mt = train_diff.view(train_diff.size(0), len(touch_pairs), 2)
                per_coord_rmse = {}
                per_coord_mse = {}
                for idx, (x_key, y_key) in enumerate(touch_pairs):
                    dx = diff_mt[:, idx, 0]
                    dy = diff_mt[:, idx, 1]
                    per_coord_rmse[x_key] = torch.sqrt(torch.mean(dx**2)).item()
                    per_coord_rmse[y_key] = torch.sqrt(torch.mean(dy**2)).item()
                    per_coord_mse[x_key] = torch.mean(dx**2).item()
                    per_coord_mse[y_key] = torch.mean(dy**2).item()
                mse_all = torch.mean(torch.sum(diff_mt**2, dim=2)).item()
                rmse_all = math.sqrt(mse_all)
                final_train_metrics = {
                    "per_coord_rmse": per_coord_rmse,
                    "per_coord_mse": per_coord_mse,
                    "rmse_all": rmse_all,
                    "mse_all": mse_all,
                }
            else:
                train_per_axis_rmse = torch.sqrt(torch.mean(train_diff**2, dim=0))
                train_euclid_rmse = torch.sqrt(torch.mean(torch.sum(train_diff**2, dim=1)))
                final_train_metrics = {
                    "rmse_x": train_per_axis_rmse[0].item(),
                    "rmse_y": train_per_axis_rmse[1].item(),
                    "rmse_z": train_per_axis_rmse[2].item(),
                    "rmse_euclid": train_euclid_rmse.item(),
                }

    print("\n" + "=" * 60)
    print("Evaluating on TEST set...")
    print("=" * 60)

    model.eval()
    test_loss_sum = test_count = 0.0
    test_all_pred, test_all_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            test_loss_sum += loss.item() * xb.size(0)
            test_count += xb.size(0)
            test_all_pred.append(pred.cpu())
            test_all_true.append(yb.cpu())

    test_mse = test_loss_sum / max(1, test_count)
    test_pred = torch.cat(test_all_pred, dim=0)
    test_true = torch.cat(test_all_true, dim=0)
    test_pred_real = torch.from_numpy(dataset.unnormalize_y(test_pred.numpy()))
    test_true_real = torch.from_numpy(dataset.unnormalize_y(test_true.numpy()))
    test_diff = test_pred_real - test_true_real

    if out_dim == 1:
        test_rmse = torch.sqrt(torch.mean(test_diff[:, 0] ** 2)).item()
        test_metrics = {"rmse": test_rmse}
    elif is_multi_touch:
        diff_mt = test_diff.view(test_diff.size(0), len(touch_pairs), 2)
        test_per_coord_rmse = {}
        test_per_coord_mse = {}
        for idx, (x_key, y_key) in enumerate(touch_pairs):
            dx = diff_mt[:, idx, 0]
            dy = diff_mt[:, idx, 1]
            test_per_coord_rmse[x_key] = torch.sqrt(torch.mean(dx**2)).item()
            test_per_coord_rmse[y_key] = torch.sqrt(torch.mean(dy**2)).item()
            test_per_coord_mse[x_key] = torch.mean(dx**2).item()
            test_per_coord_mse[y_key] = torch.mean(dy**2).item()
        test_mse_all = torch.mean(torch.sum(diff_mt**2, dim=2)).item()
        test_rmse_all = math.sqrt(test_mse_all)
        test_metrics = {
            "per_coord_rmse": test_per_coord_rmse,
            "per_coord_mse": test_per_coord_mse,
            "rmse_all": test_rmse_all,
            "mse_all": test_mse_all,
        }
    else:
        test_per_axis_rmse = torch.sqrt(torch.mean(test_diff**2, dim=0))
        test_euclid_rmse = torch.sqrt(torch.mean(torch.sum(test_diff**2, dim=1)))
        test_metrics = {
            "rmse_x": test_per_axis_rmse[0].item(),
            "rmse_y": test_per_axis_rmse[1].item(),
            "rmse_z": test_per_axis_rmse[2].item(),
            "rmse_euclid": test_euclid_rmse.item(),
        }

    summary_lines: List[str] = []

    def add_line(text: str = "") -> None:
        summary_lines.append(text)

    add_line("\n" + "=" * 60)
    add_line("FINAL TRAINING RESULTS SUMMARY")
    add_line("=" * 60)

    if out_dim == 1:
        add_line("\nTraining Set:")
        add_line(f"  MSE:  {final_train_mse:.6f}")
        add_line(f"  RMSE: {final_train_rmse:.3f} g")
        add_line("\nValidation Set:")
        add_line(f"  MSE:  {final_val_mse:.6f}")
        add_line(f"  RMSE: {final_val_rmse:.3f} g")
        add_line("\nTest Set:")
        add_line(f"  MSE:  {test_mse:.6f}")
        add_line(f"  RMSE: {test_metrics['rmse']:.3f} g")
    elif is_multi_touch:
        add_line("\nTraining Set:")
        if final_train_metrics:
            for coord in sorted(final_train_metrics["per_coord_rmse"].keys()):
                add_line(f"  RMSE ({coord}): {final_train_metrics['per_coord_rmse'][coord]:.2f} grid")
                add_line(f"  MSE  ({coord}): {final_train_metrics['per_coord_mse'][coord]:.4f} grid^2")
            add_line(f"  Overall RMSE: {final_train_metrics['rmse_all']:.2f} grid")
            add_line(f"  Overall MSE:  {final_train_metrics['mse_all']:.4f} grid^2")

        add_line("\nValidation Set:")
        if final_val_metrics:
            for coord in sorted(final_val_metrics["per_coord_rmse"].keys()):
                add_line(f"  RMSE ({coord}): {final_val_metrics['per_coord_rmse'][coord]:.2f} grid")
                add_line(f"  MSE  ({coord}): {final_val_metrics['per_coord_mse'][coord]:.4f} grid^2")
            add_line(f"  Overall RMSE: {final_val_metrics['rmse_all']:.2f} grid")
            add_line(f"  Overall MSE:  {final_val_metrics['mse_all']:.4f} grid^2")

        add_line("\nTest Set:")
        for coord in sorted(test_metrics["per_coord_rmse"].keys()):
            add_line(f"  RMSE ({coord}): {test_metrics['per_coord_rmse'][coord]:.2f} grid")
            add_line(f"  MSE  ({coord}): {test_metrics['per_coord_mse'][coord]:.4f} grid^2")
        add_line(f"  Overall RMSE: {test_metrics['rmse_all']:.2f} grid")
        add_line(f"  Overall MSE:  {test_metrics['mse_all']:.4f} grid^2")
    else:
        add_line("\nTraining Set:")
        add_line(f"  MSE:       {final_train_mse:.6f}")
        add_line(f"  RMSE (X):  {final_train_metrics['rmse_x']:.2f} grid")
        add_line(f"  RMSE (Y):  {final_train_metrics['rmse_y']:.2f} grid")
        add_line(f"  RMSE (Z):  {final_train_metrics['rmse_z']:.2f} grid")
        add_line(f"  RMSE (3D): {final_train_metrics['rmse_euclid']:.2f} grid")

        add_line("\nValidation Set:")
        add_line(f"  MSE:       {final_val_mse:.6f}")
        add_line(f"  RMSE (X):  {final_val_metrics['rmse_x']:.2f} grid")
        add_line(f"  RMSE (Y):  {final_val_metrics['rmse_y']:.2f} grid")
        add_line(f"  RMSE (Z):  {final_val_metrics['rmse_z']:.2f} grid")
        add_line(f"  RMSE (3D): {final_val_metrics['rmse_euclid']:.2f} grid")

        add_line("\nTest Set:")
        add_line(f"  MSE:       {test_mse:.6f}")
        add_line(f"  RMSE (X):  {test_metrics['rmse_x']:.2f} grid")
        add_line(f"  RMSE (Y):  {test_metrics['rmse_y']:.2f} grid")
        add_line(f"  RMSE (Z):  {test_metrics['rmse_z']:.2f} grid")
        add_line(f"  RMSE (3D): {test_metrics['rmse_euclid']:.2f} grid")

    add_line("=" * 60 + "\n")

    summary_text = "\n".join(summary_lines)

    for line in summary_lines:
        print(line)

    return model, (x_mean, x_std, y_mean, y_std), summary_text

