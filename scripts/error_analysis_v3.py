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
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    match_events,
    window_predictions_to_events,
)
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.v3 import (
    SpectrogramDataset,
    build_record_dataset,
    resolve_device,
    split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--event-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--gt-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--pred-min-duration-sec", type=float, default=0.0)
    parser.add_argument("--pred-merge-gap-sec", type=float, default=None)
    parser.add_argument(
        "--pred-span-mode",
        choices=["full", "center", "hop"],
        default="full",
    )
    parser.add_argument("--pred-center-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--plot-records", choices=["problem", "all", "none"], default="problem")
    parser.add_argument("--output-dir", default="artifacts/error_analysis/v3")
    return parser.parse_args()


def project_or_absolute(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(p)


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def effective_pred_merge_gap(args: argparse.Namespace) -> float:
    return (
        args.event_merge_gap_sec
        if args.pred_merge_gap_sec is None
        else args.pred_merge_gap_sec
    )


def predict_record_probs(
    model: torch.nn.Module,
    record_data: dict,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        SpectrogramDataset(
            record_data["spec"],
            record_data["motion"],
            record_data["labels"],
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["spec"].to(device), batch["motion"].to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probs, dtype=np.float32)


def overlapping_prob_stats(
    event: Event,
    spans: list[tuple[float, float]],
    probs: np.ndarray,
) -> tuple[float, float]:
    selected = [
        float(prob)
        for (start, end), prob in zip(spans, probs)
        if max(float(start), event.start) < min(float(end), event.end)
    ]
    if not selected:
        return 0.0, 0.0
    return float(np.max(selected)), float(np.mean(selected))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def classify_events(
    record_data: dict,
    probs: np.ndarray,
    threshold: float,
    event_iou_threshold: float,
    gt_min_duration_sec: float,
    gt_merge_gap_sec: float,
    pred_min_duration_sec: float,
    pred_merge_gap_sec: float,
    pred_span_mode: str,
    pred_center_fraction: float,
) -> tuple[list[Event], list[Event], list[tuple[int, int, float]], list[int], list[int]]:
    gt_events = binary_labels_to_events(
        record_data["record"]["cough_label"],
        sample_rate=int(record_data["record"]["fs_audio"]),
        min_duration_sec=gt_min_duration_sec,
        merge_gap_sec=gt_merge_gap_sec,
    )
    preds = (probs >= threshold).astype(int)
    pred_events = window_predictions_to_events(
        record_data["spans"],
        preds,
        min_duration_sec=pred_min_duration_sec,
        merge_gap_sec=pred_merge_gap_sec,
        span_mode=pred_span_mode,
        center_fraction=pred_center_fraction,
    )
    matches = match_events(
        gt_events,
        pred_events,
        iou_threshold=event_iou_threshold,
    )
    matched_gt = {gt_idx for gt_idx, _, _ in matches}
    matched_pred = {pred_idx for _, pred_idx, _ in matches}
    fn_indices = [idx for idx in range(len(gt_events)) if idx not in matched_gt]
    fp_indices = [idx for idx in range(len(pred_events)) if idx not in matched_pred]
    return gt_events, pred_events, matches, fn_indices, fp_indices


def add_event_error_rows(
    rows: list[dict],
    record_data: dict,
    gt_events: list[Event],
    pred_events: list[Event],
    matches: list[tuple[int, int, float]],
    fn_indices: list[int],
    fp_indices: list[int],
    probs: np.ndarray,
) -> None:
    record = record_data["record"]
    base = {
        "record_id": int(record["record_id"]),
        "filename": record["filename"],
        "activity": record["activity"],
        "context": record["context"],
    }
    for gt_idx in fn_indices:
        event = gt_events[gt_idx]
        max_prob, mean_prob = overlapping_prob_stats(
            event,
            record_data["spans"],
            probs,
        )
        rows.append(
            {
                **base,
                "error_type": "FN",
                "gt_start_sec": fmt_float(event.start),
                "gt_end_sec": fmt_float(event.end),
                "pred_start_sec": "",
                "pred_end_sec": "",
                "duration_sec": fmt_float(event.duration),
                "matched_iou": "",
                "max_probability": fmt_float(max_prob),
                "mean_probability": fmt_float(mean_prob),
            }
        )

    for pred_idx in fp_indices:
        event = pred_events[pred_idx]
        max_prob, mean_prob = overlapping_prob_stats(
            event,
            record_data["spans"],
            probs,
        )
        rows.append(
            {
                **base,
                "error_type": "FP",
                "gt_start_sec": "",
                "gt_end_sec": "",
                "pred_start_sec": fmt_float(event.start),
                "pred_end_sec": fmt_float(event.end),
                "duration_sec": fmt_float(event.duration),
                "matched_iou": "",
                "max_probability": fmt_float(max_prob),
                "mean_probability": fmt_float(mean_prob),
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["record_id"]),
            float(row["gt_start_sec"] or row["pred_start_sec"]),
            row["error_type"],
        )
    )


def event_spans_from_indices(events: list[Event], indices: list[int]) -> list[Event]:
    return [events[idx] for idx in indices]


def plot_event_bars(
    ax,
    events: list[Event],
    color: str,
    label: str,
    alpha: float = 0.8,
) -> None:
    for event in events:
        ax.broken_barh(
            [(event.start, event.duration)],
            (0.2, 0.6),
            facecolors=color,
            alpha=alpha,
            edgecolors=color,
            linewidth=1.4,
        )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.5])
    ax.set_yticklabels([label])
    ax.set_ylabel(label)


def add_event_backgrounds(
    ax,
    gt_events: list[Event],
    pred_events: list[Event],
) -> None:
    for event in gt_events:
        ax.axvspan(event.start, event.end, color="tab:red", alpha=0.10, linewidth=0)
    for event in pred_events:
        ax.axvspan(event.start, event.end, color="tab:orange", alpha=0.08, linewidth=0)


def robust_scaled(values: np.ndarray, center: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if center:
        values = values - float(np.median(values))
    scale = float(np.percentile(np.abs(values), 99))
    if scale <= 1.0e-12:
        scale = float(np.max(np.abs(values))) or 1.0
    return np.clip(values / scale, -1.0, 1.0)


def save_timeline(
    record_data: dict,
    probs: np.ndarray,
    threshold: float,
    gt_events: list[Event],
    pred_events: list[Event],
    fn_indices: list[int],
    fp_indices: list[int],
    output_path: Path,
) -> None:
    record = record_data["record"]
    fs_audio = int(record["fs_audio"])
    fs_motion = int(record["fs_motion"])
    audio_time = np.arange(len(record["pulm_bp"])) / fs_audio
    motion_time = np.arange(len(record["stretch_lp"])) / fs_motion
    spans = record_data["spans"]
    centers = np.asarray([(start + end) / 2 for start, end in spans], dtype=np.float32)
    preds = (probs >= threshold).astype(float)
    fn_events = event_spans_from_indices(gt_events, fn_indices)
    fp_events = event_spans_from_indices(pred_events, fp_indices)

    fig, axes = plt.subplots(
        8,
        1,
        figsize=(18, 13),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.2, 1.2, 1.0, 1.0, 0.65, 0.65, 0.75, 1.25],
        },
    )

    title = (
        f"Record {record['record_id']} | {record['filename']} | "
        f"{record['activity']} / {record['context']}"
    )
    sensor_axes = axes[:4]
    for ax in sensor_axes:
        add_event_backgrounds(ax, gt_events, pred_events)

    axes[0].plot(
        audio_time,
        robust_scaled(record["pulm_bp"], center=False),
        color="tab:blue",
        linewidth=0.55,
    )
    axes[0].set_ylabel("Pulm mic")
    axes[0].set_title(title)
    axes[0].set_ylim(-1.05, 1.05)

    axes[1].plot(
        audio_time,
        robust_scaled(record["amb_bp"], center=False),
        color="tab:cyan",
        linewidth=0.55,
    )
    axes[1].set_ylabel("Amb mic")
    axes[1].set_ylim(-1.05, 1.05)

    axes[2].plot(
        motion_time,
        robust_scaled(record["stretch_lp"]),
        color="tab:green",
        linewidth=0.9,
    )
    axes[2].set_ylabel("Stretch")
    axes[2].set_ylim(-1.05, 1.05)

    axes[3].plot(
        motion_time,
        robust_scaled(record["accz_lp"]),
        color="tab:brown",
        linewidth=0.9,
    )
    axes[3].set_ylabel("Acc Z")
    axes[3].set_ylim(-1.05, 1.05)
    axes[0].legend(
        handles=[
            Patch(facecolor="tab:red", alpha=0.18, label="GT region"),
            Patch(facecolor="tab:orange", alpha=0.16, label="Pred region"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    plot_event_bars(axes[4], gt_events, color="tab:red", label="GT", alpha=0.72)
    axes[4].set_title("Ground Truth Events", loc="left", fontsize=10, pad=2)
    plot_event_bars(axes[5], pred_events, color="tab:orange", label="Pred", alpha=0.72)
    axes[5].set_title("Predicted Events", loc="left", fontsize=10, pad=2)

    for event in fn_events:
        axes[6].broken_barh(
            [(event.start, event.duration)],
            (0.58, 0.28),
            facecolors="tab:red",
            edgecolors="tab:red",
            alpha=0.9,
            linewidth=1.4,
        )
    for event in fp_events:
        axes[6].broken_barh(
            [(event.start, event.duration)],
            (0.14, 0.28),
            facecolors="tab:purple",
            edgecolors="tab:purple",
            alpha=0.9,
            linewidth=1.4,
        )
    axes[6].set_ylim(0, 1)
    axes[6].set_yticks([0.28, 0.72])
    axes[6].set_yticklabels(["FP", "FN"])
    axes[6].set_ylabel("Errors")
    axes[6].set_title("Event-level Errors", loc="left", fontsize=10, pad=2)

    axes[7].plot(centers, probs, color="tab:blue", linewidth=1.2, marker="o", markersize=2)
    axes[7].fill_between(
        centers,
        threshold,
        probs,
        where=probs >= threshold,
        color="tab:orange",
        alpha=0.16,
        interpolate=True,
    )
    axes[7].step(centers, preds, where="mid", color="tab:orange", alpha=0.45, linewidth=1.1)
    axes[7].axhline(threshold, color="black", linestyle="--", linewidth=0.9)
    axes[7].set_ylabel("P(cough)")
    axes[7].set_xlabel("Time (s)")
    axes[7].set_ylim(-0.05, 1.05)

    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
        ax.set_xlim(0, max(audio_time[-1], centers[-1] if len(centers) else 0))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    pred_merge_gap_sec = effective_pred_merge_gap(args)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint.get("config") or load_config(args.config)
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])

    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    split_map = checkpoint.get("record_split")
    if split_map and args.split in split_map:
        record_ids = [int(x) for x in split_map[args.split]]
    else:
        _, val_ids, test_ids = split_records(metadata)
        selected = val_ids if args.split == "val" else test_ids
        record_ids = [int(x) for x in selected]

    model = Spec2DCoughCNN(num_classes=1).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    window_cfg = cfg["windowing"]
    pred_center_fraction = (
        float(window_cfg["center_fraction"])
        if args.pred_center_fraction is None
        else args.pred_center_fraction
    )
    spec_cfg = cfg["spectrogram"]
    output_dir = project_or_absolute(args.output_dir)
    timeline_dir = output_dir / "timelines"
    event_rows = []
    record_rows = []
    timeline_paths = []

    for record_id in record_ids:
        record_data = build_record_dataset(
            record_id,
            metadata,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        probs = predict_record_probs(
            model,
            record_data,
            batch_size=batch_size,
            device=device,
        )
        gt_events, pred_events, matches, fn_indices, fp_indices = classify_events(
            record_data,
            probs=probs,
            threshold=args.threshold,
            event_iou_threshold=args.event_iou_threshold,
            gt_min_duration_sec=args.gt_min_duration_sec,
            gt_merge_gap_sec=args.gt_merge_gap_sec,
            pred_min_duration_sec=args.pred_min_duration_sec,
            pred_merge_gap_sec=pred_merge_gap_sec,
            pred_span_mode=args.pred_span_mode,
            pred_center_fraction=pred_center_fraction,
        )

        record = record_data["record"]
        record_rows.append(
            {
                "record_id": int(record["record_id"]),
                "filename": record["filename"],
                "activity": record["activity"],
                "context": record["context"],
                "true_events": len(gt_events),
                "predicted_events": len(pred_events),
                "tp": len(matches),
                "fp": len(fp_indices),
                "fn": len(fn_indices),
            }
        )
        add_event_error_rows(
            event_rows,
            record_data,
            gt_events,
            pred_events,
            matches,
            fn_indices,
            fp_indices,
            probs,
        )

        should_plot = args.plot_records == "all" or (
            args.plot_records == "problem" and (fn_indices or fp_indices)
        )
        if should_plot:
            output_path = timeline_dir / f"record_{int(record['record_id']):03d}_timeline.png"
            save_timeline(
                record_data,
                probs=probs,
                threshold=args.threshold,
                gt_events=gt_events,
                pred_events=pred_events,
                fn_indices=fn_indices,
                fp_indices=fp_indices,
                output_path=output_path,
            )
            timeline_paths.append(output_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    record_summary_path = output_dir / "record_error_summary.csv"
    event_errors_path = output_dir / "event_errors.csv"
    summary_path = output_dir / "summary.json"

    write_csv(
        record_summary_path,
        record_rows,
        [
            "record_id",
            "filename",
            "activity",
            "context",
            "true_events",
            "predicted_events",
            "tp",
            "fp",
            "fn",
        ],
    )
    write_csv(
        event_errors_path,
        event_rows,
        [
            "record_id",
            "filename",
            "activity",
            "context",
            "error_type",
            "gt_start_sec",
            "gt_end_sec",
            "pred_start_sec",
            "pred_end_sec",
            "duration_sec",
            "matched_iou",
            "max_probability",
            "mean_probability",
        ],
    )

    totals = {
        "split": args.split,
        "threshold": args.threshold,
        "event_iou_threshold": args.event_iou_threshold,
        "event_merge_gap_sec": args.event_merge_gap_sec,
        "gt_min_duration_sec": args.gt_min_duration_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "pred_min_duration_sec": args.pred_min_duration_sec,
        "pred_merge_gap_sec": pred_merge_gap_sec,
        "pred_span_mode": args.pred_span_mode,
        "pred_center_fraction": pred_center_fraction,
        "records": len(record_rows),
        "true_events": int(sum(row["true_events"] for row in record_rows)),
        "predicted_events": int(sum(row["predicted_events"] for row in record_rows)),
        "tp": int(sum(row["tp"] for row in record_rows)),
        "fp": int(sum(row["fp"] for row in record_rows)),
        "fn": int(sum(row["fn"] for row in record_rows)),
        "timeline_count": len(timeline_paths),
        "timelines": [str(path) for path in timeline_paths],
    }
    summary_path.write_text(json.dumps(totals, indent=2), encoding="utf-8")

    print(
        f"Events: TP={totals['tp']} FP={totals['fp']} FN={totals['fn']} | "
        f"Timelines: {totals['timeline_count']}"
    )
    print(f"Saved record summary: {record_summary_path}")
    print(f"Saved event errors: {event_errors_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved timelines: {timeline_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
