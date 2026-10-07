import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m1 import M1
from m1_dataset import M1UWIQADataset


# ---------------------------------------
# Settings
# ---------------------------------------

SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
ITERATIONS_PER_EPOCH = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------
# Reproducibility
# ---------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------
# Dataset
# ---------------------------------------

train_dataset = M1UWIQADataset(
    "splits.csv",
    "train"
)

val_dataset = M1UWIQADataset(
    "splits.csv",
    "val"
)


# ---------------------------------------
# DataLoader
# ---------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------
# Model
# ---------------------------------------

model = M1().to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ---------------------------------------
# Training
# ---------------------------------------

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    model.train()

    train_loss = 0.0

    train_iterator = iter(train_loader)

    for iteration in range(ITERATIONS_PER_EPOCH):

        try:
            images, scores = next(train_iterator)

        except StopIteration:
            train_iterator = iter(train_loader)
            images, scores = next(train_iterator)

        images = images.to(device)
        scores = scores.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = criterion(
            predictions,
            scores
        )

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_train_loss = (
        train_loss / ITERATIONS_PER_EPOCH
    )


    # ---------------------------------------
    # Validation
    # ---------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, scores in val_loader:

            images = images.to(device)
            scores = scores.to(device)

            predictions = model(images)

            loss = criterion(
                predictions,
                scores
            )

            val_loss += loss.item()

    average_val_loss = (
        val_loss / len(val_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Val Loss: {average_val_loss:.4f}"
    )


    # ---------------------------------------
    # Save best model
    # ---------------------------------------

    if average_val_loss < best_val_loss:

        best_val_loss = average_val_loss

        torch.save(
            model.state_dict(),
            "m1_2x10_best.pth"
        )


print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.4f}")