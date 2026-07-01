"""The 1-D CNN architecture for HAR, shared by the training notebook and the
Streamlit app so both use exactly the same model definition.
"""

import torch.nn as nn


class HAR_CNN(nn.Module):
    """1-D CNN over time for 9-channel sensor windows of length 128."""

    def __init__(self, n_channels: int = 9, n_classes: int = 6, dropout: float = 0.3):
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
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class HAR_LSTM(nn.Module):
    """Bidirectional LSTM over time for 9-channel sensor windows.

    Accepts the same (batch, 9 channels, 128 timesteps) layout as HAR_CNN and
    transposes internally to the (batch, time, features) layout an LSTM expects.
    """

    def __init__(self, n_features: int = 9, hidden: int = 128, n_layers: int = 2,
                 n_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=n_layers, batch_first=True,
            dropout=dropout, bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_classes),   # *2 for bidirectional
        )

    def forward(self, x):
        x = x.transpose(1, 2)          # (batch, 9, 128) -> (batch, 128, 9)
        out, _ = self.lstm(x)          # (batch, 128, hidden*2)
        return self.head(out[:, -1, :])  # classify from the last timestep
