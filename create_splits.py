import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Load metadata
# -----------------------------
df = pd.read_csv("metadata.csv")

print("Total images:", len(df))


# -----------------------------
# First split:
# 70% train
# 30% temporary
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)


# -----------------------------
# Second split:
# Temporary → 15% validation
#                 15% test
# -----------------------------
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)


# -----------------------------
# Add split labels
# -----------------------------
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


# -----------------------------
# Combine
# -----------------------------
splits_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# -----------------------------
# Save
# -----------------------------
splits_df.to_csv("splits.csv", index=False)


# -----------------------------
# Print information
# -----------------------------
print("\nSplit completed!")

print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print("\nSplit percentages:")

print(
    "Train:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Validation:",
    round(len(val_df) / len(df) * 100, 2),
    "%"
)

print(
    "Test:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)

print("\nSaved as: splits.csv")


10 ITERATIONS
1 EPOCH