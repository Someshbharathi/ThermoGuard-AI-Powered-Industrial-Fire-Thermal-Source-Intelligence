import pandas as pd
from pathlib import Path

# ============================================================
# FILES
# ============================================================

from config.paths import (
    DATASETS_DIR,
    RESNET_DATA_DIR,
    SATELLITE_CANDIDATES,
)
RESNET_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels.csv"
CANDIDATES_FILE = SATELLITE_CANDIDATES
OUTPUT_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels_with_location.csv"

# ============================================================
# START
# ============================================================

print("=" * 70)
print("ADDING THERMAL LOCATION ID")
print("=" * 70)

# ============================================================
# LOAD RESNET DATA
# ============================================================

print("\nLoading ResNet dataset...")

resnet = pd.read_csv(RESNET_FILE)

print(f"ResNet rows: {len(resnet)}")

# ============================================================
# LOAD CORRECT MAPPING FILE
# ============================================================

print("\nLoading satellite candidate mapping...")

candidates = pd.read_csv(
    CANDIDATES_FILE,
    usecols=[
        "satellite_id",
        "thermal_location_id"
    ]
)

print(f"Candidate rows: {len(candidates)}")

# ============================================================
# CHECK DUPLICATES
# ============================================================

duplicate_ids = candidates["satellite_id"].duplicated().sum()

print(f"\nDuplicate satellite IDs: {duplicate_ids}")

if duplicate_ids > 0:
    print("Removing duplicate satellite IDs...")
    candidates = candidates.drop_duplicates(
        subset="satellite_id"
    )

# ============================================================
# MATCH
# ============================================================

print("\nMatching Sentinel-2 images...")

resnet = resnet.merge(
    candidates,
    on="satellite_id",
    how="left",
    validate="one_to_one"
)

# ============================================================
# RESULTS
# ============================================================

matched = resnet["thermal_location_id"].notna().sum()
missing = resnet["thermal_location_id"].isna().sum()

print("\n" + "=" * 70)
print("MATCH RESULTS")
print("=" * 70)

print(f"Total ResNet images:      {len(resnet)}")
print(f"Matched thermal location: {matched}")
print(f"Missing thermal location: {missing}")

# ============================================================
# STOP IF NOT ALL MATCHED
# ============================================================

if missing > 0:

    print("\nUnmatched satellite IDs:")

    print(
        resnet.loc[
            resnet["thermal_location_id"].isna(),
            "satellite_id"
        ].head(20).to_string(index=False)
    )

    raise SystemExit(
        "\nSTOPPED: Not all images were matched."
    )

# ============================================================
# CONVERT TYPE
# ============================================================

resnet["thermal_location_id"] = (
    resnet["thermal_location_id"]
    .astype(str)
    .str.strip()

)

# ============================================================
# LOCATION STATISTICS
# ============================================================

unique_locations = (
    resnet["thermal_location_id"].nunique()
)

print(
    f"\nUnique thermal locations represented: "
    f"{unique_locations}"
)

# ============================================================
# HIGH-CONFIDENCE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CANDIDATES")
print("=" * 70)

hc = resnet[
    resnet["high_confidence_candidate"] == 1
]

print(f"High-confidence images: {len(hc)}")

print(
    f"Unique thermal locations among them: "
    f"{hc['thermal_location_id'].nunique()}"
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(
    resnet["resnet_class"]
    .value_counts()
)

# ============================================================
# SAVE
# ============================================================

resnet.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("DATASET CREATED")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_FILE)

print(f"\nRows: {len(resnet)}")
print(f"Columns: {len(resnet.columns)}")

print("\nAdded:")
print("thermal_location_id")

print("\n✓ Used satellite_candidates.csv")
print("✓ Exact satellite_id → thermal_location_id mapping")
print("✓ Original files were NOT modified")
print("✓ Original TIFF files were NOT modified")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)