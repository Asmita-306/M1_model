import torch
import torch.nn as nn


class PatchCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            # Convert feature maps into a vector
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):

        x = self.features(x)

        # [batch, 128, 1, 1]
        x = x.view(x.size(0), -1)

        # [batch, 128]
        return x


class M1(nn.Module):

    def __init__(self):
        super().__init__()

        # One shared CNN processes all patches
        self.patch_cnn = PatchCNN()

        # Regression layer
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, patches):

        # patches shape:
        # [batch, 4, 3, 128, 128]

        batch_size = patches.size(0)
        num_patches = patches.size(1)

        # Combine batch and patch dimensions
        # [batch*4, 3, 128, 128]
        patches = patches.view(
            batch_size * num_patches,
            3,
            128,
            128
        )

        # Extract features
        features = self.patch_cnn(patches)

        # [batch*4, 128]
        features = features.view(
            batch_size,
            num_patches,
            128
        )

        # Average the patch features
        # [batch, 128]
        image_features = features.mean(dim=1)

        # Predict quality score
        score = self.regressor(image_features)

        # [batch]
        score = score.squeeze(1)

        return score


# ---------------------------------------
# Test the model
# ---------------------------------------

if __name__ == "__main__":

    model = M1()

    # Simulate a batch of 8 images,
    # each having 4 patches
    dummy_patches = torch.randn(
        8,
        4,
        3,
        128,
        128
    )

    output = model(dummy_patches)

    print("Input shape:")
    print(dummy_patches.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nPredicted scores:")
    print(output)