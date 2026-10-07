import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from m1_dataset import M1UWIQADataset
from m1 import M1


# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ---------------------------------------
# Datasets
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
# DataLoaders
# ---------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------
# Model
# ---------------------------------------

model = M1().to(device)


# ---------------------------------------
# Loss
# ---------------------------------------

criterion = nn.MSELoss()


# ---------------------------------------
# Optimizer
# ---------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ---------------------------------------
# Training settings
# ---------------------------------------

num_epochs = 20

best_val_loss = float("inf")


# ---------------------------------------
# Training loop
# ---------------------------------------

for epoch in range(num_epochs):

    # -------------------------------
    # TRAIN
    # -------------------------------

    model.train()

    train_loss = 0.0

    for patches, scores in train_loader:

        # Move data to device
        patches = patches.to(device)
        scores = scores.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(patches)

        # Calculate loss
        loss = criterion(
            predictions,
            scores
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Accumulate loss
        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -------------------------------
    # VALIDATION
    # -------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for patches, scores in val_loader:

            patches = patches.to(device)
            scores = scores.to(device)

            predictions = model(patches)

            loss = criterion(
                predictions,
                scores
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # -------------------------------
    # Print results
    # -------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # -------------------------------
    # Save best model
    # -------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "m1_best.pth"
        )

        print("  → Best model saved!")


print("\nTraining completed!")
print("Best validation loss:", best_val_loss)