import torch

from cough_analysis.data import load_metadata
from cough_analysis.models import Spec2DCoughCNN
from cough_analysis.v3 import build_dataset


def test_v3_single_record_dataset_shapes():
    metadata = load_metadata()
    X_spec, X_motion, y = build_dataset([0], metadata)

    assert X_spec.shape == (77, 2, 64, 38)
    assert X_motion.shape == (77, 2, 100)
    assert y.shape == (77,)


def test_v3_model_forward_shape():
    metadata = load_metadata()
    X_spec, X_motion, _ = build_dataset([0], metadata)
    model = Spec2DCoughCNN(num_classes=1)

    out = model(
        torch.tensor(X_spec[:2]),
        torch.tensor(X_motion[:2]),
    )

    assert tuple(out.shape) == (2,)
