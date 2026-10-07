import torch
import torch.nn as nn

from m1 import PatchCNN


class M2(nn.Module):

    def __init__(self):
        super().__init__()

        # Same shared CNN used in M1
        self.patch_cnn = PatchCNN()

        # Patch-level quality estimation
        # f_i: 128 -> q_i: 1
        self.quality_head = nn.Linear(128, 1)

        # Learnable saliency / importance estimation
        # f_i: 128 -> 64
        self.saliency_fc = nn.Linear(128, 64)

        # 64 -> importance energy e_i
        self.saliency_score = nn.Linear(64, 1)

    def forward(self, patches, return_attention=False):

        # Input:
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

        # Extract patch features
        features = self.patch_cnn(patches)

        # [batch*4, 128]
        features = features.view(
            batch_size,
            num_patches,
            128
        )

        # ---------------------------------------
        # Quality Branch
        # ---------------------------------------

        # [batch, 4, 1]
        quality = self.quality_head(features)

        # [batch, 4]
        # Quality is constrained to [0, 1]
        quality = torch.sigmoid(
            quality.squeeze(-1)
        )

        # ---------------------------------------
        # Saliency Branch
        # ---------------------------------------

        # f_i -> tanh(W_s f_i + b_s)
        saliency_features = torch.tanh(
            self.saliency_fc(features)
        )

        # [batch, 4, 1]
        energy = self.saliency_score(
            saliency_features
        )

        # [batch, 4]
        energy = energy.squeeze(-1)

        # ---------------------------------------
        # Attention / Importance
        # ---------------------------------------

        # Softmax across patches
        # α_i = exp(e_i) / sum_j exp(e_j)
        attention = torch.softmax(
            energy,
            dim=1
        )

        # ---------------------------------------
        # Adaptive Quality Pooling
        # ---------------------------------------

        # Q_hat = sum(alpha_i * q_i)
        predicted_quality = torch.sum(
            attention * quality,
            dim=1
        )

        if return_attention:
            return predicted_quality, quality, attention

        return predicted_quality


# ---------------------------------------
# Test the M2 model
# ---------------------------------------

if __name__ == "__main__":

    model = M2()

    # Simulate a batch of 8 images,
    # each having 4 patches
    dummy_patches = torch.randn(
        8,
        4,
        3,
        128,
        128
    )

    output, quality, attention = model(
        dummy_patches,
        return_attention=True
    )

    print("Input shape:")
    print(dummy_patches.shape)

    print("\nPatch quality shape:")
    print(quality.shape)

    print("\nAttention shape:")
    print(attention.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nPredicted scores:")
    print(output)

    print("\nAttention weights:")
    print(attention)

    print("\nAttention sum for each image:")
    print(attention.sum(dim=1))