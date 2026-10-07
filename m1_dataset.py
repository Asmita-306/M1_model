import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class M1UWIQADataset(Dataset):

    def __init__(self, csv_file, split):

        # Read split information
        self.data = pd.read_csv(csv_file)

        # Select train / val / test
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        # Resize every image to 256 x 256
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        print(f"{split} dataset: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def extract_patches(self, image):

        # image shape:
        # [3, 256, 256]

        patch1 = image[:, 0:128, 0:128]

        patch2 = image[:, 0:128, 128:256]

        patch3 = image[:, 128:256, 0:128]

        patch4 = image[:, 128:256, 128:256]

        # Combine patches
        patches = torch.stack([
            patch1,
            patch2,
            patch3,
            patch4
        ])

        # Shape:
        # [4, 3, 128, 128]

        return patches

    def __getitem__(self, index):

        # Get image path
        image_path = self.data.iloc[index]["image_path"]

        # Get quality score
        score = self.data.iloc[index]["score"]

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Resize + convert to tensor
        image = self.transform(image)

        # Extract 4 patches
        patches = self.extract_patches(image)

        # Convert score to tensor
        score = torch.tensor(
            float(score),
            dtype=torch.float32
        )

        return patches, score


# ---------------------------------------
# Test the dataset
# ---------------------------------------

if __name__ == "__main__":

    dataset = M1UWIQADataset(
        "splits.csv",
        "train"
    )

    patches, score = dataset[0]

    print("\nPatch shape:")
    print(patches.shape)

    print("\nScore:")
    print(score)