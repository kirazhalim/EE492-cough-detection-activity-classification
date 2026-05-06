from __future__ import annotations

import argparse
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
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from cough_analysis.config import load_config
from cough_analysis.data import load_metadata
from cough_analysis.event_metrics import (
    binary_labels_to_events,
    event_level_metrics,
    window_predictions_to_events,
)
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.paths import project_path
from cough_analysis.v3 import (
    SpectrogramDataset,
    build_dataset,
    build_record_dataset,
    resolve_device,
    split_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/v3.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--event-iou-threshold", type=float, default=0.2)
    parser.add_argument("--event-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="artifacts/evaluations/v3")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="v3_evaluation")
    parser.add_argument("--mlflow-run-name", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    return parser.parse_args()


def project_or_absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_path(path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    return torch.load(project_or_absolute(path), map_location=device)


def save_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Non-Cough", "Cough"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Non-Cough", "Cough"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("V3 Cough Detection Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def predict_arrays(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            motion = batch["motion"].to(device)
            batch_labels = batch["label"].int().cpu().numpy()
            logits = model(spec, motion)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(batch_labels.tolist())
    return np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probs)


def log_to_mlflow(
    args: argparse.Namespace,
    cfg: dict,
    report: dict,
    event_summary: dict,
    record_ids,
    output_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is not installed. Install tracking dependencies first: "
            ".venv/bin/python -m pip install -r requirements-tracking.txt"
        ) from exc

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    run_name = args.mlflow_run_name or f"evaluate_v3_{args.split}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "script": "evaluate_v3",
                "checkpoint": args.checkpoint,
                "config": args.config,
                "split": args.split,
                "threshold": args.threshold,
                "event_iou_threshold": args.event_iou_threshold,
                "event_merge_gap_sec": args.event_merge_gap_sec,
                "batch_size": batch_size,
                "device": str(device),
                "record_count": len(record_ids),
                "window_sec": float(cfg["windowing"]["window_sec"]),
                "hop_sec": float(cfg["windowing"]["hop_sec"]),
                "center_fraction": float(cfg["windowing"]["center_fraction"]),
                "n_mels": int(cfg["spectrogram"]["n_mels"]),
                "n_fft": int(cfg["spectrogram"]["n_fft"]),
                "mel_hop_length": int(cfg["spectrogram"]["hop_length"]),
            }
        )
        mlflow.log_metrics(
            {
                "window_accuracy": float(report["accuracy"]),
                "window_cough_precision": float(report["Cough"]["precision"]),
                "window_cough_recall": float(report["Cough"]["recall"]),
                "window_cough_f1": float(report["Cough"]["f1-score"]),
                "window_macro_f1": float(report["macro avg"]["f1-score"]),
                "window_weighted_f1": float(report["weighted avg"]["f1-score"]),
                "event_precision": float(event_summary["precision"]),
                "event_recall": float(event_summary["recall"]),
                "event_f1": float(event_summary["f1"]),
                "event_true_events": float(event_summary["true_events"]),
                "event_predicted_events": float(event_summary["predicted_events"]),
                "event_tp": float(event_summary["true_positive"]),
                "event_fp": float(event_summary["false_positive"]),
                "event_fn": float(event_summary["false_negative"]),
            }
        )
        for path in output_paths:
            mlflow.log_artifact(str(path), artifact_path="evaluation")


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    cfg = checkpoint.get("config") or load_config(args.config)

    split_map = checkpoint.get("record_split")
    metadata = load_metadata(project_or_absolute(cfg["data"]["metadata"]))
    if split_map and args.split in split_map:
        record_ids = split_map[args.split]
    else:
        _, val_ids, test_ids = split_records(metadata)
        record_ids = val_ids if args.split == "val" else test_ids

    window_cfg = cfg["windowing"]
    spec_cfg = cfg["spectrogram"]
    X_spec, X_motion, labels = build_dataset(
        record_ids,
        metadata,
        window_sec=float(window_cfg["window_sec"]),
        hop_sec=float(window_cfg["hop_sec"]),
        center_fraction=float(window_cfg["center_fraction"]),
        spectrogram_config=spec_cfg,
    )
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    loader = DataLoader(
        SpectrogramDataset(X_spec, X_motion, labels),
        batch_size=batch_size,
        shuffle=False,
    )

    model = Spec2DCoughCNN(num_classes=1).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    labels_np, preds_np, probs_np = predict_arrays(
        model,
        loader,
        device=device,
        threshold=args.threshold,
    )
    report = classification_report(
        labels_np,
        preds_np,
        target_names=["Non-Cough", "Cough"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels_np, preds_np, labels=[0, 1])

    output_dir = project_or_absolute(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{args.split}_classification_report.json"
    cm_path = output_dir / f"{args.split}_confusion_matrix.png"
    preds_path = output_dir / f"{args.split}_predictions.csv"
    events_path = output_dir / f"{args.split}_event_metrics.json"

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_confusion_matrix(cm, cm_path)
    with preds_path.open("w", encoding="utf-8") as f:
        f.write("label,prediction,probability\n")
        for label, pred, prob in zip(labels_np, preds_np, probs_np):
            f.write(f"{int(label)},{int(pred)},{float(prob):.8f}\n")

    total_event_counts = {
        "true_events": 0,
        "predicted_events": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    per_record_events = []
    for record_id in record_ids:
        record_data = build_record_dataset(
            int(record_id),
            metadata,
            window_sec=float(window_cfg["window_sec"]),
            hop_sec=float(window_cfg["hop_sec"]),
            center_fraction=float(window_cfg["center_fraction"]),
            spectrogram_config=spec_cfg,
        )
        record_loader = DataLoader(
            SpectrogramDataset(
                record_data["spec"],
                record_data["motion"],
                record_data["labels"],
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        _, record_preds, _ = predict_arrays(
            model,
            record_loader,
            device=device,
            threshold=args.threshold,
        )
        gt_events = binary_labels_to_events(
            record_data["record"]["cough_label"],
            sample_rate=int(record_data["record"]["fs_audio"]),
        )
        pred_events = window_predictions_to_events(
            record_data["spans"],
            record_preds,
            merge_gap_sec=args.event_merge_gap_sec,
        )
        record_metrics = event_level_metrics(
            gt_events,
            pred_events,
            iou_threshold=args.event_iou_threshold,
        )
        for key in total_event_counts:
            total_event_counts[key] += int(record_metrics[key])
        per_record_events.append(
            {
                "record_id": int(record_id),
                **record_metrics,
            }
        )

    tp = total_event_counts["true_positive"]
    fp = total_event_counts["false_positive"]
    fn = total_event_counts["false_negative"]
    event_precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    event_recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall > 0
        else 0.0
    )
    event_summary = {
        **total_event_counts,
        "precision": event_precision,
        "recall": event_recall,
        "f1": event_f1,
        "iou_threshold": args.event_iou_threshold,
        "merge_gap_sec": args.event_merge_gap_sec,
        "per_record": per_record_events,
    }
    events_path.write_text(json.dumps(event_summary, indent=2), encoding="utf-8")

    if args.mlflow:
        log_to_mlflow(
            args=args,
            cfg=cfg,
            report=report,
            event_summary=event_summary,
            record_ids=record_ids,
            output_paths=[report_path, cm_path, preds_path, events_path],
            batch_size=batch_size,
            device=device,
        )

    print(classification_report(labels_np, preds_np, target_names=["Non-Cough", "Cough"]))
    print(
        "Event-level: "
        f"P={event_precision:.3f} R={event_recall:.3f} F1={event_f1:.3f} "
        f"TP={tp} FP={fp} FN={fn}"
    )
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")
    print(f"Saved predictions: {preds_path}")
    print(f"Saved event metrics: {events_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
