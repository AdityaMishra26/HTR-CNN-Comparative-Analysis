import torch
import torch.nn as nn


class CustomCNNHTR(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNHTR, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Input: (B, 1, 64, 256)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        # Convert CNN features into a sequence
        self.feature_projection = nn.Linear(256 * 4, 256)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # Character classifier
        # +1 for CTC blank token
        self.classifier = nn.Linear(512, num_classes + 1)

    def forward(self, x):
        x = self.cnn(x)

        batch_size, channels, height, width = x.size()

        # Convert width into sequence/time dimension
        x = x.permute(0, 3, 1, 2)
        x = x.contiguous().view(
            batch_size,
            width,
            channels * height
        )

        x = self.feature_projection(x)

        # BiLSTM
        x, _ = self.lstm(x)

        # Character predictions
        x = self.classifier(x)

        return x