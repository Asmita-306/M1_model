import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m0 import M0
from m0_dataset import M0Dataset


SEED = 42

EPOCHS = 2
ITERATIONS_PER_EPOCH = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001

CSV_FILE = "splits.csv"


# Set random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Dataset
train_dataset = M0Dataset(
    CSV_FILE,
    "train"
)

val_dataset = M0Dataset(
    CSV_FILE,
    "val"
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# Model
model = M0().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


best_val_loss = float("inf")


# Training
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for iteration, (images, mos) in enumerate(train_loader):

        if iteration >= ITERATIONS_PER_EPOCH:
            break

        images = images.to(device)
        mos = mos.to(device)

        predictions = model(images)

        loss = criterion(predictions, mos)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss / ITERATIONS_PER_EPOCH
    )


    # Validation
    model.eval()

    val_loss = 0.0
    val_iterations = 0

    with torch.no_grad():

        for images, mos in val_loader:

            images = images.to(device)
            mos = mos.to(device)

            predictions = model(images)

            loss = criterion(predictions, mos)

            val_loss += loss.item()
            val_iterations += 1

    val_loss = val_loss / val_iterations


    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # Save best model
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "m0_2epochs_10iterations.pth"
        )


print("\nTraining complete.")
print(
    f"Best validation loss: {best_val_loss:.4f}"
)