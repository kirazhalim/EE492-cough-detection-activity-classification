from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from cough_analysis.data import decode_channel3, load_metadata, load_record_array
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    event_level_metrics,
    window_predictions_to_events,
)
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.preprocessing import FS_AUDIO, FS_MOTION
from cough_analysis.v3 import SpectrogramDataset, build_record_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--record-id", type=int)
    source.add_argument("--record-path")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--event-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/predictions")
    return parser.parse_args()


def project_or_absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def preprocess_external_record(record_path: Path) -> dict:
    raw = load_record_array(record_path)
    stretch, cough_label = decode_channel3(raw[:, 2])
    record = {
        "record_id": 0,
        "filename": record_path.name,
        "date": "",
        "subject": "unknown",
        "activity": "unknown",
        "context": "unknown",
        "path": str(record_path),
        "pulmonary": raw[:, 0].astype(np.float32),
        "ambient": raw[:, 1].astype(np.float32),
        "stretch": stretch,
        "accel_z": raw[:, 3].astype(np.float32),
        "cough_label": cough_label,
        "num_samples": raw.shape[0],
    }

    from scipy import signal
    from cough_analysis.preprocessing import butter_bandpass, butter_lowpass

    pulmonary = record["pulmonary"].astype(np.float64)
    ambient = record["ambient"].astype(np.float64)
    accz = record["accel_z"].astype(np.float64)
    stretch_f = record["stretch"].astype(np.float64)

    b_bp, a_bp = butter_bandpass(60, 2200, FS_AUDIO, order=4)
    pulmonary_bp = signal.filtfilt(b_bp, a_bp, pulmonary - np.median(pulmonary))
    ambient_bp = signal.filtfilt(b_bp, a_bp, ambient - np.median(ambient))
    n_motion = int(len(stretch_f) * (FS_MOTION / FS_AUDIO))
    stretch_resampled = signal.resample(stretch_f - np.median(stretch_f), n_motion)
    accz_resampled = signal.resample(accz, n_motion)
    b_lp, a_lp = butter_lowpass(20, FS_MOTION, order=4)

    return {
        **record,
        "pulm_bp": pulmonary_bp.astype(np.float32),
        "amb_bp": ambient_bp.astype(np.float32),
        "stretch_lp": signal.filtfilt(b_lp, a_lp, stretch_resampled).astype(np.float32),
        "accz_lp": signal.filtfilt(b_lp, a_lp, accz_resampled).astype(np.float32),
        "duration_sec": len(pulmonary_bp) / FS_AUDIO,
        "fs_audio": FS_AUDIO,
        "fs_motion": FS_MOTION,
    }


def save_timeline(
    record: dict,
    spans: list[tuple[float, float]],
    probs: np.ndarray,
    pred_events,
    gt_events,
    output_path: Path,
) -> None:
    time = np.arange(len(record["pulm_bp"])) / record["fs_audio"]
    audio = record["pulm_bp"]
    denom = np.max(np.abs(audio)) or 1.0
    audio_norm = audio / denom
    centers = np.asarray([(s + e) / 2 for s, e in spans])

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, audio_norm, color="gray", linewidth=0.5)
    axes[0].set_ylabel("Audio")
    axes[0].set_title(f"Record prediction: {record['filename']}")

    axes[1].fill_between(
        time,
        0,
        record["cough_label"].astype(float),
        color="silver",
        step="pre",
    )
    for event in gt_events:
        axes[1].axvspan(event.start, event.end, color="red", alpha=0.25)
    axes[1].set_ylabel("GT")
    axes[1].set_ylim(-0.05, 1.1)

    axes[2].plot(centers, probs, color="tab:blue", marker="o", markersize=2, linewidth=1)
    for event in pred_events:
        axes[2].axvspan(event.start, event.end, color="tab:orange", alpha=0.25)
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("P(cough)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint["config"]

    metadata = load_metadata(project_or_absolute(args.metadata))
    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]

    if args.record_id is not None:
        record_data = build_record_dataset(
            args.record_id,
            metadata,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
    else:
        record_path = project_or_absolute(args.record_path)
        record = preprocess_external_record(record_path)
        from cough_analysis.v3 import audio_to_log_mel, build_centered_windows, make_mel_transform

        windows = build_centered_windows(
            record,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
        )
        mel_transform = make_mel_transform(spectrogram_config=spec_cfg)
        specs = audio_to_log_mel(
            windows["audio"],
            mel_transform=mel_transform,
            log_eps=float(spec_cfg.get("log_eps", 1.0e-9)),
        )
        record_data = {
            "record": record,
            "spec": specs,
            "motion": windows["motion"],
            "labels": windows["labels"],
            "spans": windows["spans"],
        }

    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        SpectrogramDataset(
            record_data["spec"],
            record_data["motion"],
            record_data["labels"],
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["spec"].to(device), batch["motion"].to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    probs_np = np.asarray(probs)
    preds_np = (probs_np >= args.threshold).astype(int)

    gt_events = binary_labels_to_events(
        record_data["record"]["cough_label"],
        sample_rate=int(record_data["record"]["fs_audio"]),
    )
    pred_events = window_predictions_to_events(
        record_data["spans"],
        preds_np,
        merge_gap_sec=args.event_merge_gap_sec,
    )
    metrics = event_level_metrics(
        gt_events,
        pred_events,
        iou_threshold=args.event_iou_threshold,
    )

    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(record_data["record"]["filename"]).stem
    pred_csv = output_dir / f"{stem}_window_predictions.csv"
    events_json = output_dir / f"{stem}_events.json"
    timeline_png = output_dir / f"{stem}_timeline.png"

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec", "probability", "prediction", "label"])
        for (start, end), prob, pred, label in zip(
            record_data["spans"],
            probs_np,
            preds_np,
            record_data["labels"],
        ):
            writer.writerow([start, end, float(prob), int(pred), int(label)])

    events_json.write_text(
        json.dumps(
            {
                "record": record_data["record"]["filename"],
                "threshold": args.threshold,
                "event_metrics": metrics,
                "predicted_events": [event.__dict__ for event in pred_events],
                "ground_truth_events": [event.__dict__ for event in gt_events],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_timeline(
        record_data["record"],
        record_data["spans"],
        probs_np,
        pred_events,
        gt_events,
        timeline_png,
    )

    print(
        f"Predicted events: {len(pred_events)} | "
        f"GT events: {len(gt_events)} | "
        f"Event F1: {metrics['f1']:.3f}"
    )
    print(f"Saved window predictions: {pred_csv}")
    print(f"Saved events: {events_json}")
    print(f"Saved timeline: {timeline_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
