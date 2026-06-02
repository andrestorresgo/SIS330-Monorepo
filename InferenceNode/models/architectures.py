import torch
import torch.nn as nn

class HemoScout(nn.Module):
    """
    Tier 1 Model: Analyzes Plethysmography (Pleth) waves to detect hemodynamic risk.
    Input shape: (Batch, 1, 100) - 1 second at 100Hz
    """
    def __init__(self):
        super(HemoScout, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class VentGuardian(nn.Module):
    """
    Tier 1 Model: Analyzes CO2 waves to detect ventilatory risk.
    Input shape: (Batch, 1, 100) - 1 second at 100Hz
    """
    def __init__(self):
        super(VentGuardian, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class MicroTransformer(nn.Module):
    """
    Tier 2 Model: Temporal analyzer (System Brain).
    Analyzes the last 60 seconds of Tier 1 probabilities.
    Input shape: (Batch, 60, 2) - [hemo_prob, vent_prob] over 60 seconds.
    """
    def __init__(self):
        super(MicroTransformer, self).__init__()
        # Simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=2, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(2, 1)

    def forward(self, x):
        # x: (Batch, 60, 2)
        x = self.transformer(x)
        # Take the last time step for prediction
        x = x[:, -1, :]
        return self.classifier(x)
