import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class UWIQADataset(Dataset):

    def __init__(self, csv_file, split):

        # Read CSV
        self.data = pd.read_csv(csv_file)

        # Keep only requested split
        self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        print(f"{split} dataset: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Get image path
        image_path = self.data.iloc[index]["image_path"]

        # Get quality score
        score = self.data.iloc[index]["score"]

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Apply preprocessing
        image = self.transform(image)

        # Convert score to float
        score = float(score)

        return image, score


# ------------------------------------
# Test the dataset
# ------------------------------------

if __name__ == "__main__":

    train_dataset = UWIQADataset(
        "splits.csv",
        "train"
    )

    image, score = train_dataset[0]

    print("\nFirst sample:")
    print("Image shape:", image.shape)
    print("Score:", score)