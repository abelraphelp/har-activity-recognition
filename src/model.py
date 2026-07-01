"""The 1-D CNN architecture for HAR, shared by the training notebook and the
Streamlit app so both use exactly the same model definition.
"""

import torch.nn as nn


class HAR_CNN(nn.Module):
    """1-D CNN over time for 9-channel sensor windows of length 128."""

    def __init__(self, n_channels: int = 9, n_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
