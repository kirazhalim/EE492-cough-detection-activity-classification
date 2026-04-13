from pathlib import Path
import numpy as np
import pandas as pd


# ----------------------------
# basic loading
# ----------------------------
def load_metadata(metadata_path: str | Path) -> pd.DataFrame:
    metadata_path = Path(metadata_path)
    df = pd.read_csv(metadata_path)

    required_cols = [
        "record_id",
        "filename",
        "date",
        "subject",
        "activity",
        "context",
        "relative_path",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    df = df.sort_values("record_id").reset_index(drop=True)
    return df


def get_record_row(metadata: pd.DataFrame, record_id: int) -> pd.Series:
    row = metadata.loc[metadata["record_id"] == record_id]
    if len(row) == 0:
        raise ValueError(f"record_id {record_id} not found in metadata.")
    return row.iloc[0]


def resolve_record_path(
    record_row: pd.Series,
    project_root: str | Path | None = None,
) -> Path:
    rel_path = Path(record_row["relative_path"])

    if project_root is None:
        return rel_path

    project_root = Path(project_root)
    return project_root / rel_path


# ----------------------------
# channel decoding
# ----------------------------
def decode_channel3(raw_col3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    raw_col3: integer array from column 3

    returns:
        stretch_signal: decoded stretch signal
        cough_label: binary cough label
    """
    raw_col3 = raw_col3.astype(np.int64)
    cough_label = raw_col3 & 1
    stretch_signal = raw_col3 >> 1
    return stretch_signal.astype(np.float32), cough_label.astype(np.int64)


def load_record_array(
    record_path: str | Path,
    dtype=np.int64,
) -> np.ndarray:
    """
    Loads one headerless CSV record.
    Handles quoted values like "2210","2215" automatically.
    """
    record_path = Path(record_path)
    
    # Pandas tırnak içindeki verileri (quoting) varsayılan olarak mükemmel temizler
    # header=None: Dosyanda başlık satırı olmadığı için
    df = pd.read_csv(record_path, header=None, quotechar='"')
    
    # Numpy array'ine çeviriyoruz
    arr = df.to_numpy(dtype=dtype)

    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"Expected shape (N, 4), got {arr.shape} for file: {record_path}"
        )

    return arr


def load_record(
    record_id: int,
    metadata: pd.DataFrame,
    project_root: str | Path | None = None,
) -> dict:
    """
    Returns decoded signals and metadata for one record.
    """
    row = get_record_row(metadata, record_id)
    record_path = resolve_record_path(row, project_root=project_root)

    raw = load_record_array(record_path)

    pulmonary = raw[:, 0].astype(np.float32)
    ambient = raw[:, 1].astype(np.float32)
    stretch, cough_label = decode_channel3(raw[:, 2])
    accel_z = raw[:, 3].astype(np.float32)

    return {
        "record_id": int(row["record_id"]),
        "filename": row["filename"],
        "date": row["date"],
        "subject": row["subject"],
        "activity": row["activity"],
        "context": row["context"],
        "path": str(record_path),
        "pulmonary": pulmonary,
        "ambient": ambient,
        "stretch": stretch,
        "accel_z": accel_z,
        "cough_label": cough_label,
        "num_samples": raw.shape[0],
    }


# ----------------------------
# multichannel packaging
# ----------------------------
def stack_channels(record_dict: dict) -> np.ndarray:
    """
    Returns shape: (num_channels, num_samples)
    Channel order:
        0: pulmonary
        1: ambient
        2: stretch
        3: accel_z
    """
    x = np.stack(
        [
            record_dict["pulmonary"],
            record_dict["ambient"],
            record_dict["stretch"],
            record_dict["accel_z"],
        ],
        axis=0,
    )
    return x.astype(np.float32)


# ----------------------------
# windowing
# ----------------------------
def sliding_window_indices(
    signal_length: int,
    window_size: int,
    hop_size: int,
) -> list[tuple[int, int]]:
    if window_size <= 0 or hop_size <= 0:
        raise ValueError("window_size and hop_size must be positive.")
    if signal_length < window_size:
        return []

    indices = []
    start = 0
    while start + window_size <= signal_length:
        end = start + window_size
        indices.append((start, end))
        start += hop_size
    return indices


def label_window_any_positive(label_window: np.ndarray) -> int:
    return int(np.any(label_window > 0))


def label_window_by_overlap(label_window: np.ndarray, threshold: float = 0.2) -> int:
    ratio = np.mean(label_window > 0)
    return int(ratio >= threshold)


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    window_size: int,
    hop_size: int,
    label_mode: str = "any_positive",
    overlap_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """
    x shape: (C, T)
    y shape: (T,)

    returns:
        Xw shape: (num_windows, C, window_size)
        yw shape: (num_windows,)
        spans: list of (start, end)
    """
    if x.ndim != 2:
        raise ValueError(f"x must have shape (C, T), got {x.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must have shape (T,), got {y.shape}")
    if x.shape[1] != len(y):
        raise ValueError("x and y length mismatch.")

    spans = sliding_window_indices(
        signal_length=x.shape[1],
        window_size=window_size,
        hop_size=hop_size,
    )

    Xw = []
    yw = []

    for start, end in spans:
        xw = x[:, start:end]
        yw_raw = y[start:end]

        if label_mode == "any_positive":
            label = label_window_any_positive(yw_raw)
        elif label_mode == "overlap":
            label = label_window_by_overlap(yw_raw, threshold=overlap_threshold)
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        Xw.append(xw)
        yw.append(label)

    if len(Xw) == 0:
        return (
            np.empty((0, x.shape[0], window_size), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            spans,
        )

    Xw = np.stack(Xw, axis=0).astype(np.float32)
    yw = np.asarray(yw, dtype=np.int64)

    return Xw, yw, spans


# ----------------------------
# dataset building
# ----------------------------
def build_window_dataset(
    metadata: pd.DataFrame,
    project_root: str | Path | None,
    sample_rate: int = 4800,
    window_size_sec: float = 0.5,
    hop_size_sec: float = 0.1,
    label_mode: str = "any_positive",
    overlap_threshold: float = 0.2,
    record_ids: list[int] | None = None,
) -> dict:
    """
    Builds a window-level dataset from selected records.

    Returns:
        {
            "X": np.ndarray,              # (N, C, W)
            "y": np.ndarray,              # (N,)
            "record_ids": np.ndarray,     # (N,)
            "activities": np.ndarray,     # (N,)
            "contexts": np.ndarray,       # (N,)
            "spans": list[(start,end)],   # per-window spans
        }
    """
    if record_ids is None:
        record_ids = metadata["record_id"].tolist()

    window_size = int(round(window_size_sec * sample_rate))
    hop_size = int(round(hop_size_sec * sample_rate))

    X_all = []
    y_all = []
    rid_all = []
    act_all = []
    ctx_all = []
    span_all = []

    for rid in record_ids:
        rec = load_record(rid, metadata, project_root=project_root)
        x = stack_channels(rec)
        y = rec["cough_label"]

        Xw, yw, spans = make_windows(
            x=x,
            y=y,
            window_size=window_size,
            hop_size=hop_size,
            label_mode=label_mode,
            overlap_threshold=overlap_threshold,
        )

        if len(yw) == 0:
            continue

        X_all.append(Xw)
        y_all.append(yw)
        rid_all.append(np.full(len(yw), rid, dtype=np.int64))
        act_all.append(np.array([rec["activity"]] * len(yw), dtype=object))
        ctx_all.append(np.array([rec["context"]] * len(yw), dtype=object))
        span_all.extend(spans)

    if len(X_all) == 0:
        return {
            "X": np.empty((0, 4, window_size), dtype=np.float32),
            "y": np.empty((0,), dtype=np.int64),
            "record_ids": np.empty((0,), dtype=np.int64),
            "activities": np.empty((0,), dtype=object),
            "contexts": np.empty((0,), dtype=object),
            "spans": [],
        }

    return {
        "X": np.concatenate(X_all, axis=0),
        "y": np.concatenate(y_all, axis=0),
        "record_ids": np.concatenate(rid_all, axis=0),
        "activities": np.concatenate(act_all, axis=0),
        "contexts": np.concatenate(ctx_all, axis=0),
        "spans": span_all,
    }