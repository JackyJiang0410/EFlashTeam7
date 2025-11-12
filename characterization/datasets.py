import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


class SensorSpatialDataset(torch.utils.data.Dataset):
    """
    Legacy dataset pairing states.csv/sensor_post_baselines.csv for single-touch localization.
    """

    def __init__(
        self,
        states_csv: str,
        sensor_csv: str,
        z_thresh: float = 145.1,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return (
                y * torch.tensor(self.y_std, device=y.device)
                + torch.tensor(self.y_mean, device=y.device)
            )
        return y * self.y_std + self.y_mean


class SingleTouchSpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for single-touch CSVs with position_*.csv format (2D grid).
    """

    def __init__(
        self,
        data_dir: str,
        pattern: str = "position_*.csv",
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
        z_value: float = 0.0,
    ):
        csv_files = sorted(Path(data_dir).glob(pattern))
        if not csv_files:
            raise ValueError(f"No files matching {pattern} found in {data_dir}")

        all_sensors, all_positions = [], []

        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        x_pos = float(row["x_pos"])
                        y_pos = float(row["y_pos"])
                        mag_readings = [
                            float(row[f"mag{i}_{axis}"])
                            for i in range(5)
                            for axis in ("x", "y", "z")
                        ]
                    except (KeyError, ValueError):
                        continue

                    if all(abs(m) < 1e-6 for m in mag_readings):
                        continue

                    all_sensors.append(mag_readings)
                    all_positions.append([x_pos, y_pos, z_value])

        if not all_sensors:
            raise ValueError(f"No valid data found in {data_dir}")

        self.X = np.asarray(all_sensors, dtype=np.float32)
        self.Y = np.asarray(all_positions, dtype=np.float32)

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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return (
                y * torch.tensor(self.y_std, device=y.device)
                + torch.tensor(self.y_mean, device=y.device)
            )
        return y * self.y_std + self.y_mean


class MultiTouchSpatialDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-touch captures with multiple (x, y) grid pairs.
    """

    def __init__(
        self,
        data_dir: str,
        pattern: str = "multi_touch_*.csv",
        max_touches: int | None = None,
        drop_zero_rows: bool = True,
        x_mean=None,
        x_std=None,
        y_mean=None,
        y_std=None,
        normalize_x: bool = True,
        normalize_y: bool = True,
        min_force_newton: float = 1.0,
    ):
        csv_files = sorted(Path(data_dir).glob(pattern))
        if not csv_files:
            raise ValueError(f"No files matching {pattern} found in {data_dir}")

        feature_names: List[str] | None = None
        pos_pairs: List[Tuple[str, str]] = []
        mag_cols: List[str] = []
        force_key = "fz"

        def _init_columns(fieldnames: List[str]):
            nonlocal feature_names, pos_pairs, mag_cols
            if feature_names is not None:
                return
            feature_names = fieldnames

            if force_key not in fieldnames:
                raise ValueError(
                    "'fz' column required in multi-touch dataset for contact filtering "
                    f"(missing in {fieldnames})"
                )

            touch_ids = []
            for name in fieldnames:
                if not name.startswith("pos") or "_" not in name:
                    continue
                prefix, axis = name.split("_", 1)
                if axis not in {"x", "y"}:
                    continue
                suffix = prefix[3:]
                if suffix.isdigit():
                    touch_ids.append(int(suffix))

            touch_ids = sorted(set(touch_ids))
            if not touch_ids:
                raise ValueError(
                    f"Could not find any pos*_x/pos*_y columns in header: {fieldnames}"
                )

            if max_touches is not None:
                touch_ids = touch_ids[:max_touches]

            for tid in touch_ids:
                x_key = f"pos{tid}_x"
                y_key = f"pos{tid}_y"
                if x_key in fieldnames and y_key in fieldnames:
                    pos_pairs.append((x_key, y_key))

            if not pos_pairs:
                raise ValueError(
                    "Touch position columns detected but pairs could not be formed."
                )

            mag_cols.extend([c for c in fieldnames if c.startswith("mag")])
            if not mag_cols:
                raise ValueError(
                    "No magnetometer columns (mag*_*) found in dataset."
                )
            if len(mag_cols) % 3 != 0:
                raise ValueError(
                    f"Expected magnetometer axes in multiples of 3, got {len(mag_cols)}"
                )

        X_rows, Y_rows = [], []

        for csv_file in csv_files:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                _init_columns(reader.fieldnames)

                for row in reader:
                    try:
                        fz_val = float(row[force_key])
                    except (KeyError, TypeError, ValueError):
                        continue

                    if abs(fz_val) <= min_force_newton:
                        continue

                    try:
                        mags = [float(row[col]) for col in mag_cols]
                    except (KeyError, TypeError, ValueError):
                        continue

                    if drop_zero_rows and all(abs(m) < 1e-6 for m in mags):
                        continue

                    try:
                        positions = []
                        for x_key, y_key in pos_pairs:
                            positions.extend([float(row[x_key]), float(row[y_key])])
                    except (KeyError, TypeError, ValueError):
                        continue

                    X_rows.append(np.asarray(mags, dtype=np.float32))
                    Y_rows.append(np.asarray(positions, dtype=np.float32))

        if not X_rows:
            raise ValueError(
                f"No usable samples found in {len(csv_files)} files under {data_dir}"
            )

        self.X = np.vstack(X_rows)
        self.Y = np.vstack(Y_rows)

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
            self.y_mean = np.zeros(self.Y.shape[1], dtype=np.float32)
            self.y_std = np.ones(self.Y.shape[1], dtype=np.float32)

        self.touch_pairs = pos_pairs
        self.mag_cols = mag_cols
        self.output_dim = self.Y.shape[1]
        self._dataset_type = "multi_touch"
        self._data_dir = data_dir
        self._pattern = pattern
        self._max_touches = max_touches
        self._drop_zero_rows = drop_zero_rows
        self._min_force_newton = min_force_newton

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def unnormalize_y(self, y):
        if isinstance(y, torch.Tensor):
            return (
                y * torch.tensor(self.y_std, device=y.device)
                + torch.tensor(self.y_mean, device=y.device)
            )
        return y * self.y_std + self.y_mean


class TouchClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset that aggregates single-touch and multi-touch samples for classification.

    Features: magnetometer readings (mag0_x ... mag4_z) optionally normalized.
    Labels: 0 for single-touch samples, 1 for multi-touch samples.

    When subsample is provided, the dataset randomly retains only the given number of
    samples from each class (after all filtering) to avoid extreme class imbalance.
    """

    def __init__(
        self,
        single_dir: str,
        multi_dir: str,
        single_pattern: str = "position_*.csv",
        multi_pattern: str = "multi_touch_*.csv",
        drop_zero_rows: bool = True,
        min_force_newton: float = 1.0,
        subsample: int | None = None,
        x_mean=None,
        x_std=None,
        normalize_x: bool = True,
    ):
        single_files = sorted(Path(single_dir).glob(single_pattern))
        multi_files = sorted(Path(multi_dir).glob(multi_pattern))

        if not single_files:
            raise ValueError(f"No single-touch files matching {single_pattern} in {single_dir}")
        if not multi_files:
            raise ValueError(f"No multi-touch files matching {multi_pattern} in {multi_dir}")

        feats, labels = [], []

        # Parse single-touch files (no force columns)
        for csv_file in single_files:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        mags = [
                            float(row[f"mag{i}_{axis}"])
                            for i in range(5)
                            for axis in ("x", "y", "z")
                        ]
                    except (KeyError, ValueError):
                        continue

                    if drop_zero_rows and all(abs(m) < 1e-6 for m in mags):
                        continue

                    feats.append(np.asarray(mags, dtype=np.float32))
                    labels.append(0)

        # Parse multi-touch files (require normal force filter)
        for csv_file in multi_files:
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    continue
                if "fz" not in reader.fieldnames:
                    raise ValueError(
                        f"Expected 'fz' column in multi-touch file {csv_file} for force filtering."
                    )

                for row in reader:
                    try:
                        fz_val = float(row["fz"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    if abs(fz_val) <= min_force_newton:
                        continue

                    try:
                        mags = [
                            float(row[col])
                            for col in reader.fieldnames
                            if col.startswith("mag")
                        ]
                    except (KeyError, TypeError, ValueError):
                        continue

                    if len(mags) != 15:
                        continue

                    if drop_zero_rows and all(abs(m) < 1e-6 for m in mags):
                        continue

                    feats.append(np.asarray(mags, dtype=np.float32))
                    labels.append(1)

        if not feats:
            raise ValueError("No valid samples found for touch classification dataset.")

        feats = np.vstack(feats)
        labels = np.asarray(labels, dtype=np.int64)

        if subsample is not None:
            feats_balanced = []
            labels_balanced = []
            rng = np.random.default_rng(seed=0)
            for class_id in (0, 1):
                idxs = np.where(labels == class_id)[0]
                if len(idxs) > subsample:
                    idxs = rng.choice(idxs, size=subsample, replace=False)
                feats_balanced.append(feats[idxs])
                labels_balanced.append(np.full(len(idxs), class_id, dtype=np.int64))
            feats = np.vstack(feats_balanced)
            labels = np.concatenate(labels_balanced)

        self.X = feats
        self.Y = labels

        self.normalize_x = normalize_x
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

        # Metadata for re-instantiation
        self._single_dir = single_dir
        self._multi_dir = multi_dir
        self._single_pattern = single_pattern
        self._multi_pattern = multi_pattern
        self._drop_zero_rows = drop_zero_rows
        self._min_force_newton = min_force_newton
        self._subsample = subsample

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

