import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

from m0 import M0
from m0_dataset import M0Dataset


BATCH_SIZE = 8
CSV_FILE = "splits.csv"
MODEL_FILE = "m0_2epochs_10iterations.pth"


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test dataset
test_dataset = M0Dataset(
    CSV_FILE,
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Load model
model = M0().to(device)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()


# Predictions
actual_scores = []
predicted_scores = []


with torch.no_grad():

    for images, scores in test_loader:

        images = images.to(device)

        predictions = model(images)

        actual_scores.extend(
            scores.numpy()
        )

        predicted_scores.extend(
            predictions.cpu().numpy()
        )


actual_scores = np.array(actual_scores)
predicted_scores = np.array(predicted_scores)


# Metrics
plcc, _ = pearsonr(
    actual_scores,
    predicted_scores
)

srcc, _ = spearmanr(
    actual_scores,
    predicted_scores
)

krcc, _ = kendalltau(
    actual_scores,
    predicted_scores
)

rmse = np.sqrt(
    mean_squared_error(
        actual_scores,
        predicted_scores
    )
)


# Results
print("\n==============================")
print("M0 TEST RESULTS")
print("==============================")

print(f"PLCC : {plcc:.4f}")
print(f"SRCC : {srcc:.4f}")
print(f"KRCC : {krcc:.4f}")
print(f"RMSE : {rmse:.4f}")


# Sample predictions
print("\nSample predictions:")

for i in range(min(10, len(actual_scores))):

    print(
        f"Actual: {actual_scores[i]:.3f} "
        f"Predicted: {predicted_scores[i]:.3f}"
    )