import torch
import torch.nn as nn


class WidarNet(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # -----------------------------
        # CNN Feature Extractor
        # -----------------------------
        self.cnn = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.30)

        )

        # -----------------------------
        # GRU
        # -----------------------------
        self.gru = nn.GRU(

            input_size=64 * 5 * 5,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3

        )

        # -----------------------------
        # Classifier
        # -----------------------------
        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(128, num_classes)

        )

    def forward(self, x):

        # x shape:
        # [Batch, Channel, Height, Width, Time]

        B, C, H, W, T = x.shape

        # Move time dimension forward
        x = x.permute(0, 4, 1, 2, 3)

        sequence = []

        for t in range(T):

            frame = x[:, t]

            frame = self.cnn(frame)

            frame = frame.reshape(B, -1)

            sequence.append(frame)

        sequence = torch.stack(sequence, dim=1)

        output, _ = self.gru(sequence)

        output = output[:, -1, :]

        prediction = self.classifier(output)

        return prediction