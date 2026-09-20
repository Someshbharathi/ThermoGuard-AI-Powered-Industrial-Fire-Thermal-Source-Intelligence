import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SATELLITE_CANDIDATES,
)
CANDIDATES_FILE = SATELLITE_CANDIDATES

OUTPUT_FILE = RESNET_DATA_DIR / "resnet_label_table.csv"

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA)

# Only use successfully downloaded images
df = df[df["status"] == "success"].copy()

print("=" * 70)
print("CREATING RESNET LABEL TABLE")
print("=" * 70)

print("\nTotal successful Sentinel-2 images:", len(df))

# ============================================================
# STRONG INDUSTRIAL CANDIDATES
# ============================================================

industrial_labels = [
    "Industrial_Persistent",
    "Likely_Industrial"
]

industrial = df[
    df["weak_label"].isin(industrial_labels)
].copy()

print("\nStrong industrial candidates:")
print(industrial["weak_label"].value_counts())

print("Total industrial candidates:", len(industrial))

# ============================================================
# NON-INDUSTRIAL CANDIDATES
# ============================================================

natural = df[
    df["weak_label"] == "Possible_Natural"
].copy()

print("\nPossible natural candidates:", len(natural))

# We want the same number as industrial candidates
n_negative = min(len(industrial), len(natural))

natural_sample = natural.sample(
    n=n_negative,
    random_state=42
).copy()

print("Selected non-industrial candidates:", len(natural_sample))

# ============================================================
# ASSIGN BINARY LABELS
# ============================================================

industrial["resnet_label"] = 1
industrial["resnet_class"] = "Industrial"

natural_sample["resnet_label"] = 0
natural_sample["resnet_class"] = "Non_Industrial"

# ============================================================
# COMBINE
# ============================================================

selected = pd.concat(
    [industrial, natural_sample],
    ignore_index=True
)

# ============================================================
# ADD TRAINING STATUS
# ============================================================

selected["label_source"] = selected["weak_label"].apply(
    lambda x:
        "Strong_Industrial_Weak_Label"
        if x in industrial_labels
        else "Possible_Natural_Weak_Label"
)

selected["training_status"] = "candidate"

# ============================================================
# SELECT IMPORTANT COLUMNS
# ============================================================

columns = [
    "satellite_id",
    "latitude",
    "longitude",
    "firms_date",
    "firms_datetime",
    "weak_label",
    "label_confidence",
    "image_date",
    "cloud_percentage",
    "status",
    "image_path",
    "resnet_label",
    "resnet_class",
    "label_source",
    "training_status"
]

selected = selected[columns]

# ============================================================
# SHUFFLE
# ============================================================

selected = selected.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

selected.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESNET CANDIDATE DATASET")
print("=" * 70)

print("\nClass distribution:")
print(selected["resnet_class"].value_counts())

print("\nNumeric labels:")
print(selected["resnet_label"].value_counts())

print("\nOriginal weak labels:")
print(selected["weak_label"].value_counts())

print("\nCloud statistics:")
print(selected["cloud_percentage"].describe())

print("\nMissing image files:")

missing = 0

for path in selected["image_path"]:

    if not os.path.exists(path):
        missing += 1

print(missing)

print("\nOutput file:")
print(OUTPUT)

print("\nTotal candidate images:", len(selected))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)