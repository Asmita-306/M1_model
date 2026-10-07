import pandas as pd
from pathlib import Path

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

excel_file = BASE_DIR / "data" / "IQA Value.xlsx"
image_dir = BASE_DIR / "data" / "images"
output_file = BASE_DIR / "metadata.csv"


# -----------------------------
# 2. Read Excel file
# -----------------------------
df = pd.read_excel(excel_file)

print("Excel columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# -----------------------------
# 3. Check required columns
# -----------------------------
if "image" not in df.columns or "score" not in df.columns:
    raise ValueError(
        "Excel must contain columns named 'image' and 'score'."
    )


# -----------------------------
# 4. Create image filenames
# -----------------------------
df["image"] = df["image"].astype(int)

df["filename"] = df["image"].apply(
    lambda x: f"{x:04d}.png"
)


# -----------------------------
# 5. Check whether images exist
# -----------------------------
df["image_path"] = df["filename"].apply(
    lambda x: str(Path("data") / "images" / x)
)
df["exists"] = df["filename"].apply(
    lambda x: (image_dir / x).exists()
)


# -----------------------------
# 6. Display missing images
# -----------------------------
missing = df[~df["exists"]]

if len(missing) > 0:

    print("\nWARNING: Missing images:")
    print(missing[["image", "filename"]].to_string(index=False))

else:

    print("\nAll images found successfully!")


# -----------------------------
# 7. Keep only required columns
# -----------------------------
metadata = df[["image_path", "score"]].copy()


# -----------------------------
# 8. Save metadata
# -----------------------------
metadata.to_csv(output_file, index=False)

print("\nMetadata created successfully!")
print(f"Saved to: {output_file}")

print("\nFirst 10 entries:")
print(metadata.head(10))

print("\nTotal entries:", len(metadata))