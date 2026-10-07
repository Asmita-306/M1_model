import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

from m1_dataset import M1UWIQADataset
from m2 import M2


# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ---------------------------------------
# Test dataset
# ---------------------------------------

test_dataset = M1UWIQADataset(
    "splits.csv",
    "test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------
# Load M2 model
# ---------------------------------------

model = M2().to(device)

model.load_state_dict(
    torch.load(
        "m2_2x10_best.pth",
        map_location=device
    )
)

model.eval()


# ---------------------------------------
# Store predictions and actual scores
# ---------------------------------------

all_predictions = []
all_scores = []


# ---------------------------------------
# Testing
# ---------------------------------------

with torch.no_grad():

    for patches, scores in test_loader:

        patches = patches.to(device)

        predictions = model(patches)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_scores.extend(
            scores.numpy()
        )


# Convert to NumPy arrays

all_predictions = np.array(all_predictions)
all_scores = np.array(all_scores)


# ---------------------------------------
# Calculate metrics
# ---------------------------------------

plcc = pearsonr(
    all_scores,
    all_predictions
)[0]

srcc = spearmanr(
    all_scores,
    all_predictions
)[0]

krcc = kendalltau(
    all_scores,
    all_predictions
)[0]

rmse = np.sqrt(
    mean_squared_error(
        all_scores,
        all_predictions
    )
)


# ---------------------------------------
# Print results
# ---------------------------------------

print("\n==============================")
print("M2 TEST RESULTS")
print("==============================")

print(f"PLCC : {plcc:.4f}")
print(f"SRCC : {srcc:.4f}")
print(f"KRCC : {krcc:.4f}")
print(f"RMSE : {rmse:.4f}")


# ---------------------------------------
# Show a few predictions
# ---------------------------------------

print("\nSample predictions:")

for i in range(min(10, len(all_scores))):

    print(
        f"Actual: {all_scores[i]:.3f} "
        f"Predicted: {all_predictions[i]:.3f}"
    )
    