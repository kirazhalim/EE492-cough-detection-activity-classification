from __future__ import annotations

import torch
from torch import nn


class Spec2DCoughCNN(nn.Module):
    """Spectrogram audio branch and motion branch with late fusion."""

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.audio_branch = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.motion_branch = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, spec: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        xa = self.audio_branch(spec).flatten(1)
        xm = self.motion_branch(motion).flatten(1)
        out = self.classifier(torch.cat([xa, xm], dim=1))
        return out.squeeze(-1) if out.shape[-1] == 1 else out

