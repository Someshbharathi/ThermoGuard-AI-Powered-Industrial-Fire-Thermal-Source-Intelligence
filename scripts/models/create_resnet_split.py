import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from config.paths import RESNET_DATA_DIR

# ============================================================
# FILES
# ============================================================

INPUT_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels_with_location.csv"
OUTPUT_DIR = RESNET_DATA_DIR / "resnet_splits"
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATING STRATIFIED RESNET TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal images: {len(df)}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "satellite_id",
    "thermal_location_id",
    "resnet_class",
    "resnet_label",
    "high_confidence_candidate",
    "image_path"
]

missing = [c for c in required if c not in df.columns]

if missing:
    print("\nERROR: Missing columns:")
    for c in missing:
        print(f"  - {c}")
    raise SystemExit

# ============================================================
# DUPLICATE CHECK
# ============================================================

print("\nChecking duplicates...")

duplicate_satellite = df["satellite_id"].duplicated().sum()
duplicate_location = df["thermal_location_id"].duplicated().sum()

print(f"Duplicate satellite IDs: {duplicate_satellite}")
print(f"Duplicate thermal locations: {duplicate_location}")

if duplicate_satellite > 0:
    raise SystemExit("ERROR: Duplicate satellite IDs found.")

# ============================================================
# ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL CLASS DISTRIBUTION")
print("=" * 70)

print(df["resnet_class"].value_counts())

# ============================================================
# FIRST SPLIT: 80% TRAIN / 20% TEMP
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["resnet_class"],
    random_state=RANDOM_SEED
)

# ============================================================
# SECOND SPLIT: TEMP → 50% VALIDATION / 50% TEST
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["resnet_class"],
    random_state=RANDOM_SEED
)

# ============================================================
# ADD SPLIT LABEL
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

# ============================================================
# SORT FOR REPRODUCIBILITY
# ============================================================

train_df = train_df.sort_values("satellite_id")
val_df = val_df.sort_values("satellite_id")
test_df = test_df.sort_values("satellite_id")

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SAVE
# ============================================================

train_file = OUTPUT_DIR / "train.csv"
val_file = OUTPUT_DIR / "validation.csv"
test_file = OUTPUT_DIR / "test.csv"
all_file = OUTPUT_DIR / "resnet_all_splits.csv"

train_df.to_csv(train_file, index=False)
val_df.to_csv(val_file, index=False)
test_df.to_csv(test_file, index=False)

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

all_df.to_csv(all_file, index=False)

# ============================================================
# SPLIT SIZES
# ============================================================

print("\n" + "=" * 70)
print("FINAL SPLIT SIZES")
print("=" * 70)

print(f"\nTraining:   {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Testing:    {len(test_df)}")
print(f"Total:      {len(all_df)}")

# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print("\nTRAINING:")
print(train_df["resnet_class"].value_counts())

print("\nVALIDATION:")
print(val_df["resnet_class"].value_counts())

print("\nTESTING:")
print(test_df["resnet_class"].value_counts())

# ============================================================
# CLASS PERCENTAGES
# ============================================================

print("\n" + "=" * 70)
print("CLASS PERCENTAGES")
print("=" * 70)

for name, data in [
    ("TRAINING", train_df),
    ("VALIDATION", val_df),
    ("TESTING", test_df)
]:

    print(f"\n{name}")

    percentages = (
        data["resnet_class"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(percentages)

# ============================================================
# HIGH-CONFIDENCE CANDIDATES
# ============================================================

print("\n" + "=" * 70)
print("HIGH-CONFIDENCE CANDIDATES")
print("=" * 70)

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Testing", test_df)
]:

    count = int(
        data["high_confidence_candidate"].sum()
    )

    print(f"{name}: {count}")

print(
    f"Total: "
    f"{int(df['high_confidence_candidate'].sum())}"
)

# ============================================================
# THERMAL LOCATION LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("THERMAL LOCATION LEAKAGE CHECK")
print("=" * 70)

train_locations = set(
    train_df["thermal_location_id"]
)

val_locations = set(
    val_df["thermal_location_id"]
)

test_locations = set(
    test_df["thermal_location_id"]
)

train_val = train_locations & val_locations
train_test = train_locations & test_locations
val_test = val_locations & test_locations

print(
    f"Train ↔ Validation overlap: "
    f"{len(train_val)}"
)

print(
    f"Train ↔ Test overlap:       "
    f"{len(train_test)}"
)

print(
    f"Validation ↔ Test overlap:  "
    f"{len(val_test)}"
)

if (
    len(train_val) == 0
    and len(train_test) == 0
    and len(val_test) == 0
):
    print("\n✓ NO THERMAL LOCATION LEAKAGE")
else:
    print("\n✗ WARNING: THERMAL LOCATION LEAKAGE DETECTED")

# ============================================================
# IMAGE PATH CHECK
# ============================================================

print("\n" + "=" * 70)
print("IMAGE PATH CHECK")
print("=" * 70)

all_paths = pd.concat(
    [
        train_df["image_path"],
        val_df["image_path"],
        test_df["image_path"]
    ]
)

missing_paths = (
    ~all_paths.map(Path).map(Path.exists)
).sum()

print(f"Missing image files: {missing_paths}")

if missing_paths == 0:
    print("✓ All image files exist")
else:
    print("WARNING: Some image files are missing.")

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(f"\nTraining:")
print(train_file)

print(f"\nValidation:")
print(val_file)

print(f"\nTesting:")
print(test_file)

print(f"\nCombined:")
print(all_file)

print("\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)