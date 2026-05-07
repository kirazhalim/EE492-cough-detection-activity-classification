from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "cough_analysis_mpl"))

import numpy as np
import pandas as pd
from scipy import signal


FS_AUDIO = 4800
FS_MOTION = 100

METADATA_COLUMNS = [
    "record_id",
    "filename",
    "date",
    "subject",
    "activity",
    "context",
    "clothing",
    "relative_path",
]

ACTIVITY_OPTIONS = ["sitting", "standing", "walking", "running"]
CONTEXT_OPTIONS = [
    "clean",
    "coughnoise",
    "musicnoise",
    "sneezenoise",
    "snoozenoise",
    "doornoise",
    "falsepositive",
    "noise",
]
CLOTHING_OPTIONS = ["underclothes", "overclothes"]
plt = None


@dataclass
class RecordingInfo:
    record_id: int
    date_str: str
    date_iso: str
    subject: str
    activity: str
    context: str
    clothing: str

    @property
    def filename(self) -> str:
        return (
            f"{self.record_id:03d}_{self.date_str}_{self.subject}_"
            f"{self.activity}_{self.context}.csv"
        )


def project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / ".git").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find repository root.")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Preview raw sensor CSV files, ask for approval and metadata, "
            "then copy approved files into data/curated_csv and append metadata."
        )
    )
    parser.add_argument("csv_paths", nargs="*", type=Path, help="Raw CSV file(s).")
    parser.add_argument(
        "--select",
        action="store_true",
        help="Choose raw CSV file(s) with a file explorer dialog.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Process every CSV under this directory, recursively.",
    )
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=root / "data" / "curated_csv",
        help="Ready-to-use CSV output directory.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=root / "data" / "metadata.csv",
        help="Metadata CSV to append.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=root / "artifacts" / "raw_previews",
        help="Directory where preview PNGs are saved.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save preview PNGs but do not open matplotlib windows.",
    )
    parser.add_argument(
        "--no-save-preview",
        action="store_true",
        help="Do not write preview PNGs.",
    )
    return parser.parse_args()


def get_pyplot(no_show: bool):
    global plt
    if plt is None:
        if no_show:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        plt = pyplot
    return plt


def to_iso_date(date_str: str) -> str:
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


def sanitize_token(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("Empty metadata value is not allowed.")
    return value


def default_date_from_name(path: Path) -> str:
    match = re.search(r"(20\d{6})", path.stem)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d")


def choose_csv_files() -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"Could not import tkinter for file selection: {exc}")
        return []

    downloads = Path.home() / "Downloads"
    initial_dir = downloads if downloads.exists() else Path.home()

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilenames(
            parent=root,
            title="Select raw CSV file(s)",
            initialdir=str(initial_dir),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
    finally:
        root.destroy()

    return [Path(path).expanduser().resolve() for path in selected]


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        return pd.DataFrame(columns=METADATA_COLUMNS)
    metadata = pd.read_csv(metadata_path)
    missing = [column for column in METADATA_COLUMNS if column not in metadata.columns]
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    return metadata


def next_record_id(curated_dir: Path, metadata: pd.DataFrame) -> int:
    max_id = -1
    if "record_id" in metadata.columns and len(metadata):
        max_id = max(max_id, int(pd.to_numeric(metadata["record_id"]).max()))

    if curated_dir.exists():
        for path in curated_dir.glob("*.csv"):
            prefix = path.name.split("_", 1)[0]
            if prefix.isdigit():
                max_id = max(max_id, int(prefix))

    return max_id + 1


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 4:
        raise ValueError(f"Expected exactly 4 columns, found {df.shape[1]}.")

    df = df.apply(pd.to_numeric, errors="raise")
    df.columns = ["pulmonary", "ambient", "stretch_encoded", "accel_z"]

    encoded = df["stretch_encoded"].astype(np.int64).to_numpy()
    df["stretch"] = np.right_shift(encoded, 1)
    df["cough_label"] = np.bitwise_and(encoded, 1)
    return df


def normalize(values: np.ndarray) -> np.ndarray:
    centered = values - np.nanmedian(values)
    denom = np.nanmax(np.abs(centered))
    return centered / denom if denom else centered


def minmax_downsample(t: np.ndarray, y: np.ndarray, max_points: int = 120_000):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y)
    if len(y) <= max_points:
        return t, y

    bucket_count = max(max_points // 2, 1)
    bucket_width = int(np.ceil(len(y) / bucket_count))
    out_t: list[float] = []
    out_y: list[float] = []

    for start in range(0, len(y), bucket_width):
        stop = min(start + bucket_width, len(y))
        bucket = y[start:stop]
        low = start + int(np.argmin(bucket))
        high = start + int(np.argmax(bucket))
        ordered = (low, high) if low <= high else (high, low)
        for idx in ordered:
            out_t.append(float(t[idx]))
            out_y.append(float(y[idx]))

    return np.asarray(out_t), np.asarray(out_y)


def butter_bandpass(values: np.ndarray, low: float, high: float, fs: int) -> np.ndarray:
    b, a = signal.butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, values)


def butter_lowpass(values: np.ndarray, cutoff: float, fs: int) -> np.ndarray:
    b, a = signal.butter(4, cutoff / (fs / 2), btype="low")
    return signal.filtfilt(b, a, values)


def build_preview(df: pd.DataFrame, title: str, pyplot):
    pulmonary = df["pulmonary"].to_numpy(dtype=np.float64)
    ambient = df["ambient"].to_numpy(dtype=np.float64)
    stretch = df["stretch"].to_numpy(dtype=np.float64)
    accel_z = df["accel_z"].to_numpy(dtype=np.float64)
    label = df["cough_label"].to_numpy(dtype=np.float64)

    pulmonary_f = butter_bandpass(pulmonary - np.median(pulmonary), 60, 2200, FS_AUDIO)
    ambient_f = butter_bandpass(ambient - np.median(ambient), 60, 2200, FS_AUDIO)

    motion_len = max(2, int(round(len(stretch) * FS_MOTION / FS_AUDIO)))
    stretch_resampled = signal.resample(stretch - np.median(stretch), motion_len)
    accel_resampled = signal.resample(accel_z - np.median(accel_z), motion_len)
    stretch_f = butter_lowpass(stretch_resampled, 20, FS_MOTION)
    accel_f = butter_lowpass(accel_resampled, 20, FS_MOTION)

    duration = len(pulmonary) / FS_AUDIO
    t_audio = np.linspace(0, duration, len(pulmonary), endpoint=False)
    t_motion = np.linspace(0, duration, len(stretch_f), endpoint=False)

    fig, axes = pyplot.subplots(5, 1, figsize=(14, 10), dpi=130, sharex=True)
    fig.suptitle(f"Raw CSV Preview: {title}", fontsize=12)

    plot_specs = [
        (axes[0], t_audio, pulmonary, pulmonary_f, "Pulmonary mic", "tab:blue"),
        (axes[1], t_audio, ambient, ambient_f, "Ambient mic", "tab:red"),
        (axes[2], t_audio, stretch, stretch_f, "Stretch sensor", "tab:purple"),
        (axes[3], t_audio, accel_z, accel_f, "Accelerometer Z", "tab:green"),
    ]

    for ax, raw_t, raw_y, filtered_y, label_text, color in plot_specs:
        tx, raw_plot = minmax_downsample(raw_t, normalize(raw_y))
        if len(filtered_y) == len(raw_y):
            tf = raw_t
        else:
            tf = t_motion
        tf, filtered_plot = minmax_downsample(tf, normalize(filtered_y))
        ax.plot(tx, raw_plot, color="0.72", linewidth=0.75, label="raw")
        ax.plot(tf, filtered_plot, color=color, linewidth=1.0, label="filtered")
        ax.set_title(label_text, fontsize=10)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1, 0, 1])
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right", fontsize=8)

    tx, label_plot = minmax_downsample(t_audio, label)
    axes[4].fill_between(
        tx,
        0,
        label_plot,
        step="pre",
        color="0.75",
        edgecolor="0.35",
        linewidth=0.5,
    )
    axes[4].set_title("Ground truth cough label", fontsize=10)
    axes[4].set_ylim(0, 1.1)
    axes[4].set_yticks([0, 1])
    axes[4].set_xlabel("Time (s)")
    axes[4].grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    return fig


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            answer = input(f"{prompt} {suffix}: ").strip().lower()
        except EOFError:
            print()
            return default
        if not answer:
            return default
        if answer in {"y", "yes", "e", "evet"}:
            return True
        if answer in {"n", "no", "h", "hayir", "hayır"}:
            return False
        print("Please answer yes/no.")


def prompt_text(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        return default or ""
    return value or (default or "")


def prompt_choice(prompt: str, options: list[str], default: str) -> str:
    option_text = ", ".join(options)
    while True:
        value = sanitize_token(prompt_text(f"{prompt} ({option_text})", default))
        if value in options:
            return value
        print(f"Invalid value. Allowed options: {option_text}")


def ask_recording_info(path: Path, record_id: int) -> RecordingInfo:
    while True:
        date_str = prompt_text("Recording date YYYYMMDD", default_date_from_name(path))
        date_str = date_str.strip()
        if re.fullmatch(r"\d{8}", date_str):
            break
        print("Date must be in YYYYMMDD format.")

    subject = sanitize_token(prompt_text("Subject name", "halim"))
    activity = prompt_choice("Activity", ACTIVITY_OPTIONS, "sitting")
    context = prompt_choice("Condition/context", CONTEXT_OPTIONS, "clean")
    clothing = prompt_choice("Clothing", CLOTHING_OPTIONS, "underclothes")

    return RecordingInfo(
        record_id=record_id,
        date_str=date_str,
        date_iso=to_iso_date(date_str),
        subject=subject,
        activity=activity,
        context=context,
        clothing=clothing,
    )


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [path.expanduser().resolve() for path in args.csv_paths]
    if args.raw_dir:
        paths.extend(sorted(args.raw_dir.expanduser().resolve().rglob("*.csv")))
    if args.select or (not paths and not args.raw_dir):
        paths.extend(choose_csv_files())

    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        if path.suffix.lower() != ".csv":
            print(f"Skipping non-CSV file: {path}")
            continue
        unique_paths.append(path)
    return unique_paths


def append_metadata(metadata_path: Path, row: dict) -> None:
    header = not metadata_path.exists()
    pd.DataFrame([row], columns=METADATA_COLUMNS).to_csv(
        metadata_path,
        mode="a",
        header=header,
        index=False,
    )


def process_one(path: Path, args: argparse.Namespace, metadata: pd.DataFrame) -> bool:
    print(f"\n=== {path.name} ===")
    df = read_raw_csv(path)

    pyplot = get_pyplot(args.no_show)
    fig = build_preview(df, path.name, pyplot)
    preview_path = None
    if not args.no_save_preview:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = args.preview_dir / f"{path.stem}_preview.png"
        fig.savefig(preview_path, bbox_inches="tight")
        print(f"Preview saved: {preview_path}")

    if not args.no_show:
        pyplot.show(block=True)
    pyplot.close(fig)

    if not prompt_yes_no("Do these 4 sensor plots and GT label look correct?"):
        print("Not approved; file was not added to curated dataset.")
        return False

    record_id = next_record_id(args.curated_dir, metadata)
    info = ask_recording_info(path, record_id)
    destination = args.curated_dir / info.filename
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")

    args.curated_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    append_metadata(
        args.metadata_path,
        {
            "record_id": info.record_id,
            "filename": info.filename,
            "date": info.date_iso,
            "subject": info.subject,
            "activity": info.activity,
            "context": info.context,
            "clothing": info.clothing,
            "relative_path": f"curated_csv/{info.filename}",
        },
    )

    print(f"Added curated file: {destination}")
    if preview_path:
        print(f"Kept preview image: {preview_path}")
    return True


def main() -> None:
    args = parse_args()
    args.curated_dir = args.curated_dir.expanduser().resolve()
    args.metadata_path = args.metadata_path.expanduser().resolve()
    args.preview_dir = args.preview_dir.expanduser().resolve()

    csv_paths = discover_inputs(args)
    if not csv_paths:
        raise SystemExit("No CSV files given. Pass file paths or --raw-dir.")

    metadata = load_metadata(args.metadata_path)
    approved = 0
    for path in csv_paths:
        try:
            if process_one(path, args, metadata):
                approved += 1
                metadata = load_metadata(args.metadata_path)
        except Exception as exc:
            print(f"Error while processing {path}: {exc}")

    print(f"\nDone. Added {approved} of {len(csv_paths)} CSV file(s).")


if __name__ == "__main__":
    main()
