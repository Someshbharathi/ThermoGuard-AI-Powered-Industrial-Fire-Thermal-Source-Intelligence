import pandas as pd
from pathlib import Path
from config.paths import RESNET_DATA_DIR

# ============================================================
# FILES
# ============================================================

SPLIT_DIR = RESNET_DATA_DIR / "resnet_splits"
QUALITY_REPORT = RESNET_DATA_DIR / "resnet_full_quality_report.csv"
OUTPUT_DIR = RESNET_DATA_DIR / "resnet_final_manifests"

# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING FINAL RESNET DATASET MANIFESTS")
print("=" * 70)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD QUALITY REPORT
# ============================================================

print("\nLoading quality report...")

quality = pd.read_csv(QUALITY_FILE)

print(
    f"Quality report rows: {len(quality)}"
)

# Check required columns
required_quality_columns = [
    "satellite_id",
    "quality_status",
    "quality_reason",
    "bad_pixel_percent",
    "cloud_percentage"
]

missing_columns = [
    col for col in required_quality_columns
    if col not in quality.columns
]

if missing_columns:
    raise SystemExit(
        f"ERROR: Quality report is missing columns: "
        f"{missing_columns}"
    )

quality_info = quality[
    required_quality_columns
].copy()

# Make sure satellite IDs have the same type
quality_info["satellite_id"] = (
    quality_info["satellite_id"]
    .astype(str)
    .str.strip()
)

# Check duplicate IDs
duplicate_quality_ids = (
    quality_info["satellite_id"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate satellite IDs in quality report: "
    f"{duplicate_quality_ids}"
)

if duplicate_quality_ids > 0:
    raise SystemExit(
        "ERROR: Duplicate satellite IDs found "
        "in quality report."
    )

# ============================================================
# PROCESS SPLITS
# ============================================================

split_files = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv"
}

final_datasets = {}

for split_name, split_file in split_files.items():

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load split
    # --------------------------------------------------------

    split_df = pd.read_csv(split_file)

    print(
        f"Original images: {len(split_df)}"
    )

    # --------------------------------------------------------
    # Standardize satellite ID
    # --------------------------------------------------------

    split_df["satellite_id"] = (
        split_df["satellite_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove old quality columns
    # --------------------------------------------------------

    old_quality_columns = [
        "quality_status",
        "quality_reason",
        "bad_pixel_percent",
        "cloud_percentage"
    ]

    for col in old_quality_columns:

        if col in split_df.columns:

            split_df = split_df.drop(
                columns=[col]
            )

    # --------------------------------------------------------
    # Merge ACTUAL quality scan results
    # --------------------------------------------------------

    split_df = split_df.merge(
        quality_info,
        on="satellite_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Verify quality information
    # --------------------------------------------------------

    missing_quality = (
        split_df["quality_status"]
        .isna()
        .sum()
    )

    print(
        f"Missing quality records: "
        f"{missing_quality}"
    )

    if missing_quality > 0:

        missing_ids = split_df.loc[
            split_df["quality_status"].isna(),
            "satellite_id"
        ].tolist()

        print(
            "Missing IDs:",
            missing_ids[:20]
        )

        raise SystemExit(
            "ERROR: Some images have no "
            "quality information."
        )

    # --------------------------------------------------------
    # Quality breakdown
    # --------------------------------------------------------

    print("\nQuality breakdown:")

    print(
        split_df["quality_status"]
        .value_counts()
    )

    # --------------------------------------------------------
    # KEEP
    # --------------------------------------------------------

    final_df = split_df[
        split_df["quality_status"] == "KEEP"
    ].copy()

    # --------------------------------------------------------
    # REVIEW
    # --------------------------------------------------------

    review_df = split_df[
        split_df["quality_status"] == "REVIEW"
    ].copy()

    # --------------------------------------------------------
    # EXCLUDE
    # --------------------------------------------------------

    exclude_df = split_df[
        split_df["quality_status"] == "EXCLUDE"
    ].copy()

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    final_file = (
        OUTPUT_DIR /
        f"{split_name}_final.csv"
    )

    review_file = (
        OUTPUT_DIR /
        f"{split_name}_review.csv"
    )

    exclude_file = (
        OUTPUT_DIR /
        f"{split_name}_excluded.csv"
    )

    final_df.to_csv(
        final_file,
        index=False
    )

    review_df.to_csv(
        review_file,
        index=False
    )

    exclude_df.to_csv(
        exclude_file,
        index=False
    )

    final_datasets[split_name] = final_df

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        f"\nFinal usable images: "
        f"{len(final_df)}"
    )

    print(
        f"Review images: "
        f"{len(review_df)}"
    )

    print(
        f"Excluded images: "
        f"{len(exclude_df)}"
    )

# ============================================================
# COMBINE FINAL DATASET
# ============================================================

train_final = final_datasets["train"]
validation_final = final_datasets["validation"]
test_final = final_datasets["test"]

all_final = pd.concat(
    [
        train_final,
        validation_final,
        test_final
    ],
    ignore_index=True
)

all_file = (
    OUTPUT_DIR /
    "all_final.csv"
)

all_final.to_csv(
    all_file,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(
    f"\nTraining:   {len(train_final)}"
)

print(
    f"Validation: {len(validation_final)}"
)

print(
    f"Testing:    {len(test_final)}"
)

print(
    f"TOTAL:      {len(all_final)}"
)

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLASS DISTRIBUTION")
print("=" * 70)

for name, df in [
    ("TRAINING", train_final),
    ("VALIDATION", validation_final),
    ("TESTING", test_final),
    ("ALL", all_final)
]:

    print(f"\n{name}:")

    print(
        df["resnet_label"]
        .value_counts()
    )

# ============================================================
# HIGH CONFIDENCE
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CANDIDATES")
print("=" * 70)

for name, df in [
    ("Training", train_final),
    ("Validation", validation_final),
    ("Testing", test_final)
]:

    count = int(
        df["high_confidence_candidate"]
        .sum()
    )

    print(
        f"{name}: {count}"
    )

print(
    f"Total: "
    f"{int(all_final['high_confidence_candidate'].sum())}"
)

# ============================================================
# IMAGE PATH CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE PATH CHECK")
print("=" * 70)

missing_paths = 0

for path in all_final["image_path"]:

    if not Path(path).exists():
        missing_paths += 1

print(
    f"Missing image files: "
    f"{missing_paths}"
)

if missing_paths > 0:
    print(
        "WARNING: Some image files are missing."
    )

# ============================================================
# SPLIT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("SPLIT LEAKAGE CHECK")
print("=" * 70)

train_ids = set(
    train_final["satellite_id"]
)

val_ids = set(
    validation_final["satellite_id"]
)

test_ids = set(
    test_final["satellite_id"]
)

print(
    "Train ∩ Validation:",
    len(train_ids & val_ids)
)

print(
    "Train ∩ Test:",
    len(train_ids & test_ids)
)

print(
    "Validation ∩ Test:",
    len(val_ids & test_ids)
)

# Thermal location leakage
train_locations = set(
    train_final["thermal_location_id"]
)

val_locations = set(
    validation_final["thermal_location_id"]
)

test_locations = set(
    test_final["thermal_location_id"]
)

print(
    "\nThermal location leakage:"
)

print(
    "Train ∩ Validation:",
    len(train_locations & val_locations)
)

print(
    "Train ∩ Test:",
    len(train_locations & test_locations)
)

print(
    "Validation ∩ Test:",
    len(val_locations & test_locations)
)

# ============================================================
# FILES CREATED
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "\nOriginal TIFF images were NOT modified."
)

print(
    "Original train/validation/test split was preserved."
)

print(
    "Only KEEP images are included in the final model manifests."
)