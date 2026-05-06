import numpy as np

from cough_analysis.event_metrics import (
    Event,
    binary_labels_to_events,
    event_iou,
    event_level_metrics,
    window_predictions_to_events,
)


def test_binary_labels_to_events():
    labels = np.array([0, 1, 1, 0, 1, 0])
    events = binary_labels_to_events(labels, sample_rate=2)
    assert events == [Event(0.5, 1.5), Event(2.0, 2.5)]


def test_window_predictions_to_events_merges_overlaps():
    spans = [(0.0, 1.0), (0.5, 1.5), (2.0, 3.0)]
    preds = np.array([1, 1, 1])
    events = window_predictions_to_events(spans, preds)
    assert events == [Event(0.0, 1.5), Event(2.0, 3.0)]


def test_event_metrics_counts_matches():
    gt = [Event(0.0, 1.0), Event(3.0, 4.0)]
    pred = [Event(0.1, 1.1), Event(5.0, 6.0)]
    metrics = event_level_metrics(gt, pred, iou_threshold=0.2)
    assert metrics["true_positive"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["f1"] == 0.5
    assert event_iou(gt[0], pred[0]) > 0.8

