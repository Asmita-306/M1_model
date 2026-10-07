import torch
import torch.nn as nn


class M0(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        score = self.regressor(x)

        return score.squeeze(1)


if __name__ == "__main__":
    model = M0()

    dummy_images = torch.randn(8, 3, 224, 224)

    output = model(dummy_images)

    print("Input shape:", dummy_images.shape)
    print("Output shape:", output.shape)
    print("Predictions:", output)