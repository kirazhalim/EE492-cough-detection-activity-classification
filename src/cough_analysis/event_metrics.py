from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Event:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def binary_labels_to_events(
    labels: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.0,
) -> list[Event]:
    labels = np.asarray(labels).astype(bool)
    events = []
    start_idx = None

    for idx, active in enumerate(labels):
        if active and start_idx is None:
            start_idx = idx
        elif not active and start_idx is not None:
            event = Event(start_idx / sample_rate, idx / sample_rate)
            if event.duration >= min_duration_sec:
                events.append(event)
            start_idx = None

    if start_idx is not None:
        event = Event(start_idx / sample_rate, len(labels) / sample_rate)
        if event.duration >= min_duration_sec:
            events.append(event)

    return events


def window_predictions_to_events(
    spans: list[tuple[float, float]],
    predictions: np.ndarray,
    min_duration_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
) -> list[Event]:
    predictions = np.asarray(predictions).astype(bool)
    active_spans = [
        (float(start), float(end))
        for (start, end), active in zip(spans, predictions)
        if active
    ]
    if not active_spans:
        return []

    events = []
    current_start, current_end = active_spans[0]
    for start, end in active_spans[1:]:
        if start <= current_end + merge_gap_sec:
            current_end = max(current_end, end)
        else:
            event = Event(current_start, current_end)
            if event.duration >= min_duration_sec:
                events.append(event)
            current_start, current_end = start, end

    event = Event(current_start, current_end)
    if event.duration >= min_duration_sec:
        events.append(event)
    return events


def event_iou(a: Event, b: Event) -> float:
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union > 0 else 0.0


def match_events(
    gt_events: list[Event],
    pred_events: list[Event],
    iou_threshold: float = 0.2,
) -> list[tuple[int, int, float]]:
    candidates = []
    for gt_idx, gt_event in enumerate(gt_events):
        for pred_idx, pred_event in enumerate(pred_events):
            iou = event_iou(gt_event, pred_event)
            if iou >= iou_threshold:
                candidates.append((gt_idx, pred_idx, iou))

    candidates.sort(key=lambda x: x[2], reverse=True)
    matched_gt = set()
    matched_pred = set()
    matches = []

    for gt_idx, pred_idx, iou in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches


def event_level_metrics(
    gt_events: list[Event],
    pred_events: list[Event],
    iou_threshold: float = 0.2,
) -> dict:
    matches = match_events(gt_events, pred_events, iou_threshold=iou_threshold)
    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    return {
        "true_events": len(gt_events),
        "predicted_events": len(pred_events),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "iou_threshold": iou_threshold,
    }

