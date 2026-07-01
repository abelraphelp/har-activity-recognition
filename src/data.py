"""Reusable loaders for the UCI HAR Dataset.

Two representations are provided:
- load_features():    the 561 pre-computed features  -> (X, y, subjects)
- load_raw_signals(): the raw 9-channel signals       -> (n, 128, 9)

Both are used across the analysis and modeling notebooks so the loading logic
lives in one place.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# The 9 raw inertial signal channels, in a fixed order.
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def get_data_dir() -> Path:
    """Locate 'UCI HAR Dataset' whether called from the project root or a subdir."""
    for base in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent):
        candidate = base / "data" / "UCI HAR Dataset"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "UCI HAR Dataset not found. Run: python src/download_data.py"
    )


def load_activity_map(data_dir: Path | None = None) -> dict[int, str]:
    """Return {1: 'WALKING', ..., 6: 'LAYING'}."""
    data_dir = data_dir or get_data_dir()
    labels = pd.read_csv(
        data_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"]
    )
    return dict(zip(labels["id"], labels["name"]))


def load_features(split: str = "train", data_dir: Path | None = None):
    """Load the 561-feature representation.

    Returns (X, y, subjects):
        X        : ndarray (n_windows, 561)
        y        : ndarray (n_windows,)  activity ids 1-6
        subjects : ndarray (n_windows,)  subject ids
    """
    data_dir = data_dir or get_data_dir()
    X = pd.read_csv(data_dir / f"{split}/X_{split}.txt", sep=r"\s+", header=None).values
    y = pd.read_csv(data_dir / f"{split}/y_{split}.txt", header=None).values.ravel()
    subjects = pd.read_csv(
        data_dir / f"{split}/subject_{split}.txt", header=None
    ).values.ravel()
    return X, y, subjects


def load_raw_signals(split: str = "train", data_dir: Path | None = None) -> np.ndarray:
    """Load the raw 9-channel signals as an array of shape (n_windows, 128, 9)."""
    data_dir = data_dir or get_data_dir()
    sig_dir = data_dir / split / "Inertial Signals"
    channels = [
        pd.read_csv(sig_dir / f"{name}_{split}.txt", sep=r"\s+", header=None).values
        for name in SIGNAL_NAMES
    ]
    return np.stack(channels, axis=-1)
