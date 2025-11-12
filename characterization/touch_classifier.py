import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from characterization.datasets import TouchClassificationDataset  # type: ignore
    from characterization.models import MLP  # type: ignore
else:
    from .datasets import TouchClassificationDataset
    from .models import MLP

BASE_DATABASE_DIR = Path(__file__).resolve().parents[1] / "Database"
BASE_DATA_DIR = BASE_DATABASE_DIR / "Data"
BASE_RESULT_DIR = BASE_DATABASE_DIR / "result"


def resolve_dataset_path(path_like: str) -> Path:
    path = Path(path_like).expanduser()
    if path.exists():
        return path

    replaced = Path(str(path).replace(os.sep + "Data" + os.sep, os.sep + "Database" + os.sep + "Data" + os.sep))
    if replaced != path and replaced.exists():
        return replaced

    relative_candidate = BASE_DATA_DIR / Path(path_like)
    if relative_candidate.exists():
        return relative_candidate

    name_candidate = BASE_DATA_DIR / Path(path_like).name
    if name_candidate.exists():
        return name_candidate

    raise FileNotFoundError(f"Dataset folder not found: {path_like}\n"
                            f"Tried: {path}, {replaced}, {relative_candidate}, {name_candidate}")


def split_indices(n: int, test_split: float = 0.15, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("Dataset too small for train/val/test splits (need at least 3 samples).")

    idxs = np.arange(n)
    np.random.shuffle(idxs)

    test_size = max(1, int(test_split * n))
    remaining = n - test_size
    if remaining < 2:
        test_size = max(1, n - 2)
        remaining = n - test_size

    val_size = max(1, int(val_ratio * remaining))
    train_size = remaining - val_size
    if train_size <= 0:
        train_size = 1
        val_size = remaining - train_size

    train_idx = idxs[:train_size]
    val_idx = idxs[train_size : train_size + val_size]
    test_idx = idxs[train_size + val_size :]

    return train_idx, val_idx, test_idx


def fit_touch_classifier(
    dataset_full: TouchClassificationDataset,
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
    train_idx, val_idx, test_idx = split_indices(n, test_split=test_split)
    print(f"Data splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    x_mean = dataset_full.X[train_idx].mean(axis=0)
    x_std = dataset_full.X[train_idx].std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)

    dataset = TouchClassificationDataset(
        single_dir=dataset_full._single_dir,
        multi_dir=dataset_full._multi_dir,
        single_pattern=dataset_full._single_pattern,
        multi_pattern=dataset_full._multi_pattern,
        drop_zero_rows=dataset_full._drop_zero_rows,
        min_force_newton=dataset_full._min_force_newton,
        subsample=dataset_full._subsample,
        x_mean=x_mean,
        x_std=x_std,
        normalize_x=True,
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

    model = MLP(in_dim=dataset.X.shape[1], out_dim=2, hidden=128).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    pbar = tqdm(range(1, epochs + 1), desc="Training classifier", ncols=150)
    for epoch in pbar:
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.long().to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = train_correct / max(1, train_total)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.float().to(device)
                yb = yb.long().to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / max(1, val_total)
        val_acc = val_correct / max(1, val_total)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        pbar.set_postfix(
            {
                "Train Loss": f"{train_loss:.4f}",
                "Train Acc": f"{train_acc * 100:.1f}%",
                "Val Loss": f"{val_loss:.4f}",
                "Val Acc": f"{val_acc * 100:.1f}%",
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.float().to(device)
            yb = yb.long().to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            test_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == yb).sum().item()
            test_total += xb.size(0)

    test_loss = test_loss_sum / max(1, test_total)
    test_acc = test_correct / max(1, test_total)

    summary_lines = [
        "\n" + "=" * 60,
        "FINAL TRAINING RESULTS SUMMARY",
        "=" * 60,
        "",
        f"Final Train Loss: {train_loss:.4f}",
        f"Final Train Acc:  {train_acc * 100:.2f}%",
        "",
        f"Validation Loss:  {val_loss:.4f}",
        f"Validation Acc:   {val_acc * 100:.2f}%",
        f"Best Val Acc:     {best_val_acc * 100:.2f}%",
        "",
        f"Test Loss:        {test_loss:.4f}",
        f"Test Accuracy:    {test_acc * 100:.2f}%",
        "=" * 60 + "\n",
    ]

    for line in summary_lines:
        print(line)

    summary_text = "\n".join(summary_lines)

    return model, (x_mean.astype(np.float32), x_std.astype(np.float32)), test_acc, summary_text


def resolve_device(device_str: str | None) -> torch.device:
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no GPU is available.")
    return device


def parse_args():
    parser = argparse.ArgumentParser(description="Train touch classifier (single vs multi touch).")
    parser.add_argument(
        "--single-dir",
        type=str,
        required=True,
        help="Directory with single-touch CSV files (e.g., Database/Data/single_touch_data/).",
    )
    parser.add_argument(
        "--multi-dir",
        type=str,
        required=True,
        help="Directory with multi-touch CSV files (e.g., Database/Data/multi_touch_data/).",
    )
    parser.add_argument("--single-pattern", type=str, default="position_*.csv")
    parser.add_argument("--multi-pattern", type=str, default="multi_touch_*.csv")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument(
        "--min-force",
        type=float,
        default=1.0,
        help="Minimum normal force (N) for multi-touch samples to be included.",
    )
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="Keep rows where all magnetometer readings are approximately zero.",
    )
    parser.add_argument(
        "--class-limit",
        type=int,
        default=None,
        help="Optional maximum samples per class after filtering (for balancing).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: e.g., 'cuda', 'cpu', or 'auto' (default).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory for run artifacts (default: Database/result/...).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    single_root = resolve_dataset_path(args.single_dir)
    multi_root = resolve_dataset_path(args.multi_dir)
    dataset_full = TouchClassificationDataset(
        single_dir=str(single_root),
        multi_dir=str(multi_root),
        single_pattern=args.single_pattern,
        multi_pattern=args.multi_pattern,
        drop_zero_rows=not args.keep_zero,
        min_force_newton=args.min_force,
        subsample=args.class_limit,
        normalize_x=False,
    )

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"\n{'=' * 60}")
    print("Starting TOUCH CLASSIFICATION training")
    print(f"Single-touch samples: {np.sum(dataset_full.Y == 0)}")
    print(f"Multi-touch samples:  {np.sum(dataset_full.Y == 1)}")
    print(f"Input dim: {dataset_full.X.shape[1]}")
    print(f"{'=' * 60}\n")

    model, stats, test_acc, summary_text = fit_touch_classifier(
        dataset_full=dataset_full,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
        test_split=args.test_split,
    )

    base_result_dir = Path(args.output_dir).expanduser() if args.output_dir else BASE_RESULT_DIR
    base_result_dir.mkdir(parents=True, exist_ok=True)
    run_label = "classifier_single_multi"
    run_dir = base_result_dir / f"{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": stats[0],
            "x_std": stats[1],
            "test_accuracy": test_acc,
            "single_dir": str(single_root),
            "multi_dir": str(multi_root),
            "single_pattern": args.single_pattern,
            "multi_pattern": args.multi_pattern,
            "min_force": args.min_force,
            "drop_zero_rows": not args.keep_zero,
        },
        checkpoint_path,
    )
    (run_dir / "summary.txt").write_text(summary_text.lstrip("\n"))

    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()

