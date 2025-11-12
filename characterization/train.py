import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from characterization.datasets import (  # type: ignore
        MultiTouchSpatialDataset,
        SingleTouchSpatialDataset,
        SensorSpatialDataset,
    )
    from characterization.training import fit  # type: ignore
else:
    from .datasets import (
        MultiTouchSpatialDataset,
        SingleTouchSpatialDataset,
        SensorSpatialDataset,
    )
    from .training import fit

BASE_DATABASE_DIR = Path(__file__).resolve().parents[1] / "Database"
BASE_DATA_DIR = BASE_DATABASE_DIR / "Data"
BASE_RESULT_DIR = BASE_DATABASE_DIR / "result"


def resolve_dataset_path(path_like: str) -> Path:
    """
    Resolve dataset path, supporting legacy references to Data/... as well as new Database/Data/... layout.
    """
    original = Path(path_like).expanduser()
    if original.exists():
        return original

    # Try replacing '/Data/' with '/Database/Data/'
    replaced_str = str(original).replace(os.sep + "Data" + os.sep, os.sep + "Database" + os.sep + "Data" + os.sep)
    replaced = Path(replaced_str)
    if replaced != original and replaced.exists():
        return replaced

    # Try relative to BASE_DATA_DIR
    relative_candidate = BASE_DATA_DIR / Path(path_like)
    if relative_candidate.exists():
        return relative_candidate

    # As a last resort, append the final component to BASE_DATA_DIR
    name_candidate = BASE_DATA_DIR / Path(path_like).name
    if name_candidate.exists():
        return name_candidate

    raise FileNotFoundError(f"Dataset folder not found: {path_like}\n"
                            f"Tried: {original}, {replaced}, {relative_candidate}, {name_candidate}")


def build_dataset(args):
    if args.mode == "single_touch":
        data_root = resolve_dataset_path(args.folder)
        dataset = SingleTouchSpatialDataset(
            data_dir=str(data_root),
            pattern=args.pattern,
            z_value=args.z_value,
            normalize_x=False,
            normalize_y=False,
        )
        dataset._dataset_type = "single_touch"
        dataset._data_dir = str(data_root)
        dataset._pattern = args.pattern
        dataset._z_value = args.z_value
        out_dim = 3
        
    elif args.mode == "spatial":
        states_csv = os.path.join(args.folder, "states.csv")
        sensor_csv = os.path.join(args.folder, "sensor_post_baselines.csv")
        
        states_path = resolve_dataset_path(states_csv)
        sensor_path = resolve_dataset_path(sensor_csv)

        dataset = SensorSpatialDataset(
            states_csv=str(states_path),
            sensor_csv=str(sensor_path),
            z_thresh=args.z_thresh,
            normalize_x=False,
            normalize_y=False,
        )
        dataset._dataset_type = "spatial"
        dataset._data_dir = str(states_path.parent)
        dataset._states_csv = str(states_path)
        dataset._sensor_csv = str(sensor_path)
        dataset._z_thresh = args.z_thresh
        out_dim = 3

    else:  # multi_touch
        data_root = resolve_dataset_path(args.folder)
        dataset = MultiTouchSpatialDataset(
            data_dir=str(data_root),
            pattern=args.multi_touch_pattern,
            max_touches=args.multi_touch_max_touches,
            drop_zero_rows=not args.multi_touch_keep_zero,
            min_force_newton=args.multi_touch_min_force,
            normalize_x=False,
            normalize_y=False,
        )
        dataset._dataset_type = "multi_touch"
        dataset._data_dir = str(data_root)
        dataset._pattern = args.multi_touch_pattern
        dataset._max_touches = args.multi_touch_max_touches
        dataset._drop_zero_rows = not args.multi_touch_keep_zero
        dataset._min_force_newton = args.multi_touch_min_force
        out_dim = dataset.output_dim

    return dataset, out_dim


def resolve_device(device_str: str | None) -> torch.device:
    """Return torch.device, preferring CUDA when available.

    Parameters
    ----------
    device_str: Optional textual device specifier. If None or 'auto',
                choose 'cuda' when available, otherwise 'cpu'.
    """
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no GPU is available.")
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description="eFlesh localization training (spatial | single_touch | multi_touch)"
    )
    parser.add_argument(
        "--mode",
        choices=["spatial", "single_touch", "multi_touch"],
        default="single_touch",
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the dataset folder (e.g., Database/Data/local_sin_3*3/).",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    # single_touch / spatial options
    parser.add_argument("--pattern", type=str, default="position_*.csv")
    parser.add_argument("--z_thresh", type=float, default=145.1)
    parser.add_argument("--z_value", type=float, default=0.0)

    # multi-touch options
    parser.add_argument("--multi_touch_pattern", type=str, default="multi_touch_*.csv")
    parser.add_argument("--multi_touch_max_touches", type=int, default=None)
    parser.add_argument(
        "--multi_touch_keep_zero",
        action="store_true",
        help="Keep rows where magnetometer readings are all ~0",
    )
    parser.add_argument(
        "--multi_touch_min_force",
        type=float,
        default=1.0,
        help="Minimum normal force (N) to keep a sample",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: e.g., 'cuda', 'cpu', or 'auto' (default).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset, out_dim = build_dataset(args)

    print(f"\n{'=' * 60}")
    print("Starting LOCALIZATION training")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.folder}")
    print(f"Samples: {len(dataset.X)}")
    print(f"Input dim: {dataset.X.shape[1]}, Output dim: {out_dim}")
    print(f"{'=' * 60}\n")

    model, stats, summary_text = fit(
        dataset_full=dataset,
        out_dim=out_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        seed=args.seed,
    )

    BASE_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    run_label = "localization_multi" if args.mode == "multi_touch" else "localization_single"
    run_dir = BASE_RESULT_DIR / f"{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.pt"
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

    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary_text.lstrip("\n"))

    print(f"\n{'=' * 60}")
    print(f"Artifacts saved to: {run_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

