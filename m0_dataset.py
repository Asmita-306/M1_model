import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class M0Dataset(Dataset):

    def __init__(self, csv_file, split):

        self.data = pd.read_csv(csv_file)

        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"{split} dataset: {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        score = self.data.iloc[index]["score"]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        score = torch.tensor(
            float(score),
            dtype=torch.float32
        )

        return image, score


if __name__ == "__main__":

    dataset = M0Dataset("splits.csv", "train")

    image, score = dataset[0]

    print("\nImage shape:")
    print(image.shape)

    print("\nScore:")
    print(score)