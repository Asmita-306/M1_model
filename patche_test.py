import torch
from dataset import UWIQADataset


def extract_patches(image):

    # image shape = [3, 256, 256]

    patch1 = image[:, 0:128, 0:128]
    patch2 = image[:, 0:128, 128:256]
    patch3 = image[:, 128:256, 0:128]
    patch4 = image[:, 128:256, 128:256]

    patches = torch.stack([
        patch1,
        patch2,
        patch3,
        patch4
    ])

    return patches


# --------------------------------
# Load one training image
# --------------------------------

dataset = UWIQADataset(
    "splits.csv",
    "train"
)

image, score = dataset[0]

# --------------------------------
# Extract patches
# --------------------------------

patches = extract_patches(image)


# --------------------------------
# Print results
# --------------------------------

print("\nOriginal image:")
print(image.shape)

print("\nPatches:")
print(patches.shape)

print("\nScore:")
print(score)