import pandas as pd
import os

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    DATASETS_DIR,
    RESNET_DATA_DIR,
    SATELLITE_CANDIDATES,
)
CANDIDATES_FILE = SATELLITE_CANDIDATES

OUTPUT_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels.csv"
SENTINEL2_METADATA = DATASETS_DIR / "sentinel2_metadata.csv"

# ============================================================
# LOAD SENTINEL-2 METADATA
# ============================================================

df = pd.read_csv(METADATA)

# Only successful downloads
df = df[df["status"] == "success"].copy()

print("=" * 70)
print("CREATING FULL RESNET DATASET")
print("=" * 70)

print("\nSuccessful Sentinel-2 images:", len(df))

# ============================================================
# CONVERT WEAK LABEL → RESNET CLASS
# ============================================================

def assign_resnet_class(label):

    if label in [
        "Possible_Industrial",
        "Likely_Industrial",
        "Industrial_Persistent"
    ]:
        return "Industrial"

    elif label == "Likely_Mining":
        return "Mining"

    elif label == "Possible_Natural":
        return "Natural"

    elif label == "Unknown":
        return "Other_Uncertain"

    else:
        return "Other_Uncertain"


df["resnet_class"] = df["weak_label"].apply(
    assign_resnet_class
)

# Numeric class labels
class_mapping = {
    "Natural": 0,
    "Industrial": 1,
    "Mining": 2,
    "Other_Uncertain": 3
}

df["resnet_label"] = df["resnet_class"].map(
    class_mapping
)

# ============================================================
# MARK HIGH-CONFIDENCE CANDIDATES
# ============================================================

high_confidence_labels = [
    "Industrial_Persistent",
    "Likely_Industrial"
]

df["high_confidence_candidate"] = (
    df["weak_label"]
    .isin(high_confidence_labels)
    .astype(int)
)

# ============================================================
# QUALITY STATUS
# ============================================================

# Default all images to usable.
df["quality_status"] = "AVAILABLE"

# ============================================================
# IMPORTANT INFORMATION COLUMNS
# ============================================================

columns = [
    "satellite_id",
    "latitude",
    "longitude",
    "firms_date",
    "firms_datetime",
    "weak_label",
    "label_confidence",
    "resnet_class",
    "resnet_label",
    "high_confidence_candidate",
    "image_date",
    "cloud_percentage",
    "status",
    "quality_status",
    "image_path"
]

df = df[columns]

# ============================================================
# CHECK FILES
# ============================================================

missing = []

for path in df["image_path"]:

    if not os.path.exists(path):
        missing.append(path)

print("\nMissing image files:", len(missing))

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("RESNET CLASS DISTRIBUTION")
print("=" * 70)

print(
    df["resnet_class"].value_counts()
)

# ============================================================
# WEAK LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL WEAK LABEL DISTRIBUTION")
print("=" * 70)

print(
    df["weak_label"].value_counts()
)

# ============================================================
# HIGH-CONFIDENCE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CANDIDATES")
print("=" * 70)

print(
    df["high_confidence_candidate"].value_counts()
)

# ============================================================
# CLASS × CONFIDENCE
# ============================================================

print("\n" + "=" * 70)
print("CLASS × LABEL CONFIDENCE")
print("=" * 70)

print(
    pd.crosstab(
        df["resnet_class"],
        df["label_confidence"]
    )
)

# ============================================================
# CLOUD STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CLOUD STATISTICS")
print("=" * 70)

print(
    df["cloud_percentage"].describe()
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("DATASET CREATED")
print("=" * 70)

print("\nOutput:")
print(OUTPUT)

print("\nTotal images:", len(df))

print("\nClass counts:")
print(
    df["resnet_class"].value_counts()
)

print("\nOriginal Sentinel-2 TIFF files were NOT modified.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)