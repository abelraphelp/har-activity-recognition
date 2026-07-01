"""Loaders and windowing for the PAMAP2 dataset.

PAMAP2 is raw continuous 100 Hz recording (54 columns, no header). We keep the
recommended acc(+-16g) + gyro channels from all three IMUs (hand, chest, ankle),
drop heart rate / temperature / magnetometer / orientation, downsample to 50 Hz,
and segment into fixed windows.

The windowing supports a *transition-aware* mode used to test whether windows that
straddle activity boundaries (or the dataset's transient label 0) hurt the model.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# 12 protocol activities (activity id 0 = transient/between-activities).
ACTIVITY_MAP = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuuming", 17: "ironing", 24: "rope_jumping",
}

# 0-indexed columns: acc(+-16g) x/y/z and gyro x/y/z for each of the 3 IMUs.
# IMU blocks start at col 3 (hand), 20 (chest), 37 (ankle); within a block:
#   +1..+3 = acc16g, +7..+9 = gyro.
_IMU_STARTS = [3, 20, 37]
SELECTED_COLS = [s + off for s in _IMU_STARTS for off in (1, 2, 3, 7, 8, 9)]  # 18 channels
ACTIVITY_COL = 1


def load_subject(path: Path, downsample: int = 2):
    """Return (X, y): X shape (n, 18) channels, y shape (n,) activity ids (0=transient).

    NaN gaps in the IMU channels are interpolated; data is decimated by `downsample`
    (2 -> 100 Hz to 50 Hz).
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    y = df[ACTIVITY_COL].values.astype(int)
    X = df[SELECTED_COLS].astype(float)
    X = X.interpolate().bfill().ffill().values      # fill small sensor dropouts
    if downsample > 1:
        X, y = X[::downsample], y[::downsample]
    return X, y


def make_windows(X, y, win: int = 128, step: int = 64, transition_aware: bool = True):
    """Slide a window over one subject's stream.

    transition_aware=True : keep a window only if it is entirely ONE activity and not
                            transient (0) -> clean, single-activity windows.
    transition_aware=False: keep every window with a non-transient majority label
                            (majority vote) -> includes boundary-straddling windows.
    """
    windows, labels = [], []
    for start in range(0, len(X) - win + 1, step):
        seg = y[start:start + win]
        if transition_aware:
            if seg[0] != 0 and np.all(seg == seg[0]):
                windows.append(X[start:start + win]); labels.append(seg[0])
        else:
            vals, counts = np.unique(seg, return_counts=True)
            maj = vals[counts.argmax()]
            if maj != 0:
                windows.append(X[start:start + win]); labels.append(maj)
    return np.asarray(windows), np.asarray(labels)


def get_protocol_dir() -> Path:
    for base in (Path.cwd(), Path.cwd().parent, Path(__file__).resolve().parent.parent):
        p = base / "data" / "PAMAP2_Dataset" / "Protocol"
        if p.exists():
            return p
    raise FileNotFoundError("PAMAP2 not found. Run: python src/download_pamap2.py")
