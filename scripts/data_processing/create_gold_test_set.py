import pandas as pd
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    VERIFICATION_DATA_DIR,
)
INPUT_FILE = RESNET_DATA_DIR / "resnet_final_manifests" / "test_final.csv"

OUTPUT_DIR = VERIFICATION_DATA_DIR / "gold_test_set"

OUTPUT_FILE = OUTPUT_DIR / "gold_test_173.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




# ============================================================
# LOAD TEST SET
# ============================================================

print("Loading final test manifest...")

df = pd.read_csv(TEST_MANIFEST)

print(f"Original test rows: {len(df)}")


# ============================================================
# CHECKS
# ============================================================

required_columns = [
    "satellite_id",
    "thermal_location_id",
    "latitude",
    "longitude",
    "firms_date",
    "firms_datetime",
    "weak_label",
    "label_confidence",
    "image_date",
    "cloud_percentage",
    "image_path"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ============================================================
# CREATE GOLD TEST MANIFEST
# ============================================================

gold = df[required_columns].copy()


# Independent verification fields
gold["verified_label"] = ""
gold["verification_confidence"] = ""
gold["verification_basis"] = ""
gold["reviewer_notes"] = ""


# ============================================================
# QUALITY / DUPLICATE CHECKS
# ============================================================

print("\nRunning checks...")

print("Duplicate satellite IDs:",
      gold["satellite_id"].duplicated().sum())

print("Duplicate thermal locations:",
      gold["thermal_location_id"].duplicated().sum())

print("Missing image paths:",
      gold["image_path"].isna().sum())


# Check whether image files exist
missing_files = 0

for path in gold["image_path"]:
    if not Path(path).exists():
        missing_files += 1

print("Image files missing:", missing_files)


# ============================================================
# SAVE
# ============================================================

gold.to_csv(OUTPUT_FILE, index=False)

print("\n==========================================")
print("GOLD TEST SET CREATED")
print("==========================================")
print(f"Rows: {len(gold)}")
print(f"Output: {OUTPUT_FILE}")

print("\nWeak-label distribution:")
print(gold["weak_label"].value_counts())

print("\nVerification columns:")
print([
    "verified_label",
    "verification_confidence",
    "verification_basis",
    "reviewer_notes"
])