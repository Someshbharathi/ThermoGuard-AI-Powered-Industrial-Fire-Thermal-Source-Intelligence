import os
import pandas as pd
import numpy as np


# ============================================================
# THERMOGUARD
# XGBOOST DATASET CREATION
# ============================================================

print("=" * 70)
print("THERMOGUARD - XGBOOST DATASET CREATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SATELLITE_CANDIDATES,
    XGBOOST_DATA_DIR,
)
CANDIDATES_FILE = SATELLITE_CANDIDATES
FINAL_MANIFEST_DIR = RESNET_DATA_DIR / "resnet_final_manifests"
OUTPUT_DIR = XGBOOST_DATA_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading satellite candidates...")

candidates = pd.read_csv(CANDIDATES)

print("Candidate rows:", len(candidates))


print("\nLoading ResNet final manifests...")

train = pd.read_csv(
    os.path.join(
        FINAL_MANIFEST_DIR,
        "train_final.csv"
    )
)

validation = pd.read_csv(
    os.path.join(
        FINAL_MANIFEST_DIR,
        "validation_final.csv"
    )
)

test = pd.read_csv(
    os.path.join(
        FINAL_MANIFEST_DIR,
        "test_final.csv"
    )
)

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    # Identity
    "satellite_id",
    "thermal_location_id",

    # Time
    "acq_datetime",

    # Thermal
    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "confidence_score",

    # Spatial distances
    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km",

    # Spatial proximity
    "industrial_area_within_1km",
    "industrial_area_within_5km",
    "industrial_area_within_10km",
    "industrial_area_within_25km",

    "industrial_works_within_1km",
    "industrial_works_within_5km",
    "industrial_works_within_10km",
    "industrial_works_within_25km",

    "storage_tank_within_1km",
    "storage_tank_within_5km",
    "storage_tank_within_10km",
    "storage_tank_within_25km",

    "petroleum_well_within_1km",
    "petroleum_well_within_5km",
    "petroleum_well_within_10km",
    "petroleum_well_within_25km",

    "quarry_within_1km",
    "quarry_within_5km",
    "quarry_within_10km",
    "quarry_within_25km",

    "mining_within_1km",
    "mining_within_5km",
    "mining_within_10km",
    "mining_within_25km",

    # Persistence
    "detection_count",
    "active_days",
    "duration_days",
    "detections_per_active_day",
    "persistence_ratio",
    "detections_7d",
    "detections_30d",
    "detections_90d",

    # Event
    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",

    # Weak label
    "weak_label",
    "label_confidence"
]


missing_columns = [
    column
    for column in required_columns
    if column not in candidates.columns
]

if missing_columns:

    print("\nERROR: Missing columns in satellite_candidates.csv:")

    for column in missing_columns:
        print(" -", column)

    raise SystemExit(1)


print("\nAll required columns found.")


# ============================================================
# PREPARE IDS
# ============================================================

candidates["satellite_id"] = (
    candidates["satellite_id"]
    .astype(str)
    .str.strip()
)

train["satellite_id"] = (
    train["satellite_id"]
    .astype(str)
    .str.strip()
)

validation["satellite_id"] = (
    validation["satellite_id"]
    .astype(str)
    .str.strip()
)

test["satellite_id"] = (
    test["satellite_id"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CHECK DUPLICATE CANDIDATES
# ============================================================

duplicate_count = candidates["satellite_id"].duplicated().sum()

print(
    "\nDuplicate candidate satellite IDs:",
    duplicate_count
)

if duplicate_count > 0:

    raise SystemExit(
        "ERROR: Duplicate satellite IDs found."
    )


# ============================================================
# XGBOOST FEATURES
# ============================================================

features = [

    # --------------------------------------------------------
    # THERMAL FEATURES
    # --------------------------------------------------------

    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "confidence_score",


    # --------------------------------------------------------
    # SPATIAL DISTANCE FEATURES
    # --------------------------------------------------------

    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km",


    # --------------------------------------------------------
    # SPATIAL PROXIMITY FEATURES
    # --------------------------------------------------------

    # Industrial area
    "industrial_area_within_1km",
    "industrial_area_within_5km",
    "industrial_area_within_10km",
    "industrial_area_within_25km",

    # Industrial works
    "industrial_works_within_1km",
    "industrial_works_within_5km",
    "industrial_works_within_10km",
    "industrial_works_within_25km",

    # Storage tanks
    "storage_tank_within_1km",
    "storage_tank_within_5km",
    "storage_tank_within_10km",
    "storage_tank_within_25km",

    # Petroleum wells
    "petroleum_well_within_1km",
    "petroleum_well_within_5km",
    "petroleum_well_within_10km",
    "petroleum_well_within_25km",

    # Quarry
    "quarry_within_1km",
    "quarry_within_5km",
    "quarry_within_10km",
    "quarry_within_25km",

    # Mining
    "mining_within_1km",
    "mining_within_5km",
    "mining_within_10km",
    "mining_within_25km",


    # --------------------------------------------------------
    # PERSISTENCE FEATURES
    # --------------------------------------------------------

    "detection_count",
    "active_days",
    "duration_days",
    "detections_per_active_day",
    "persistence_ratio",
    "detections_7d",
    "detections_30d",
    "detections_90d",


    # --------------------------------------------------------
    # EVENT FEATURES
    # --------------------------------------------------------

    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",


    # --------------------------------------------------------
    # TEMPORAL FEATURES
    # --------------------------------------------------------

    "month",
    "day_of_year",
    "hour"
]


print(
    "\nNumber of XGBoost features:",
    len(features)
)


# ============================================================
# CREATE SPLIT FUNCTION
# ============================================================

def create_xgboost_split(manifest, split_name):

    print("\n" + "-" * 70)

    print(
        "Creating XGBoost",
        split_name,
        "dataset..."
    )


    # ========================================================
    # MERGE
    # ========================================================

    merged = manifest[
        [
            "satellite_id",
            "thermal_location_id",
            "resnet_class",
            "resnet_label",
            "weak_label",
            "label_confidence",
            "high_confidence_candidate"
        ]
    ].merge(
        candidates,
        on="satellite_id",
        how="left",
        suffixes=("_manifest", "")
    )


    print(
        "Rows after merge:",
        len(merged)
    )


    # ========================================================
    # CHECK MERGE
    # ========================================================

    missing_features = merged["frp"].isna().sum()

    print(
        "Rows missing candidate features:",
        missing_features
    )

    if missing_features > 0:

        raise SystemExit(
            "ERROR: Some ResNet samples could not be "
            "matched with candidate features."
        )


    # ========================================================
    # THERMAL LOCATION CHECK
    # ========================================================

    manifest_locations = (
        merged["thermal_location_id_manifest"]
        .astype(str)
        .str.strip()
    )

    candidate_locations = (
        merged["thermal_location_id"]
        .astype(str)
        .str.strip()
    )

    mismatch = (
        manifest_locations != candidate_locations
    ).sum()


    print(
        "Thermal location mismatches:",
        mismatch
    )


    if mismatch > 0:

        raise SystemExit(
            "ERROR: Thermal location mismatch."
        )


    # ========================================================
    # TARGET
    # ========================================================

    merged["target"] = merged["resnet_class"]


    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    datetime_values = pd.to_datetime(
        merged["acq_datetime"],
        errors="coerce"
    )


    invalid_datetime = datetime_values.isna().sum()

    print(
        "Invalid datetime values:",
        invalid_datetime
    )


    if invalid_datetime > 0:

        raise SystemExit(
            "ERROR: Invalid acquisition datetime."
        )


    merged["month"] = datetime_values.dt.month

    merged["day_of_year"] = (
        datetime_values.dt.dayofyear
    )

    merged["hour"] = datetime_values.dt.hour


    # ========================================================
    # SELECT FEATURES
    # ========================================================

    result = merged[
        features
    ].copy()


    # ========================================================
    # CONVERT FEATURES TO NUMERIC
    # ========================================================

    for column in features:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce"
        )


    # ========================================================
    # CHECK MISSING VALUES
    # ========================================================

    missing_values = (
        result[features]
        .isna()
        .sum()
        .sum()
    )


    print(
        "Missing feature values:",
        missing_values
    )


    if missing_values > 0:

        print(
            "\nColumns containing missing values:"
        )

        for column in features:

            count = result[column].isna().sum()

            if count > 0:

                print(
                    f" - {column}: {count}"
                )


        raise SystemExit(
            "ERROR: Missing feature values detected."
        )


    # ========================================================
    # CHECK INFINITE VALUES
    # ========================================================

    infinite_values = np.isinf(
        result[features].to_numpy()
    ).sum()


    print(
        "Infinite feature values:",
        infinite_values
    )


    if infinite_values > 0:

        raise SystemExit(
            "ERROR: Infinite feature values detected."
        )


    # ========================================================
    # ADD TARGET
    # ========================================================

    result["target"] = merged["target"].values


    # ========================================================
    # ADD METADATA
    # ========================================================

    metadata = merged[
        [
            "satellite_id",
            "thermal_location_id",
            "latitude",
            "longitude",
            "acq_date",
            "acq_datetime",
            "source_satellite",
            "weak_label",
            "label_confidence",
            "resnet_label",
            "high_confidence_candidate"
        ]
    ].copy()


    # Put metadata first, followed by features and target.

    output = pd.concat(
        [
            metadata,
            result[features],
            result[["target"]]
        ],
        axis=1
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        split_name + ".csv"
    )


    output.to_csv(
        output_path,
        index=False
    )


    print(
        "Saved:",
        output_path
    )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print("\nTarget distribution:")

    print(
        output["target"]
        .value_counts()
        .sort_index()
    )


    return output


# ============================================================
# CREATE TRAIN DATASET
# ============================================================

train_xgb = create_xgboost_split(
    train,
    "train"
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

validation_xgb = create_xgboost_split(
    validation,
    "validation"
)


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_xgb = create_xgboost_split(
    test,
    "test"
)


# ============================================================
# COMBINE ALL
# ============================================================

all_xgb = pd.concat(
    [
        train_xgb,
        validation_xgb,
        test_xgb
    ],
    ignore_index=True
)


all_path = os.path.join(
    OUTPUT_DIR,
    "all.csv"
)


all_xgb.to_csv(
    all_path,
    index=False
)


print(
    "\nSaved:",
    all_path
)


# ============================================================
# LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)


train_ids = set(
    train_xgb["satellite_id"]
)

validation_ids = set(
    validation_xgb["satellite_id"]
)

test_ids = set(
    test_xgb["satellite_id"]
)


train_locations = set(
    train_xgb["thermal_location_id"]
)

validation_locations = set(
    validation_xgb["thermal_location_id"]
)

test_locations = set(
    test_xgb["thermal_location_id"]
)


print(
    "Train ↔ Validation satellite overlap:",
    len(train_ids & validation_ids)
)

print(
    "Train ↔ Test satellite overlap:",
    len(train_ids & test_ids)
)

print(
    "Validation ↔ Test satellite overlap:",
    len(validation_ids & test_ids)
)


print(
    "Train ↔ Validation thermal-location overlap:",
    len(train_locations & validation_locations)
)

print(
    "Train ↔ Test thermal-location overlap:",
    len(train_locations & test_locations)
)

print(
    "Validation ↔ Test thermal-location overlap:",
    len(validation_locations & test_locations)
)


# ============================================================
# DATASET SIZES
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)


print(
    "Train:",
    len(train_xgb)
)

print(
    "Validation:",
    len(validation_xgb)
)

print(
    "Test:",
    len(test_xgb)
)

print(
    "Total:",
    len(all_xgb)
)

print(
    "Features:",
    len(features)
)


# ============================================================
# FEATURE LIST
# ============================================================

print("\nXGBoost feature list:")

for number, feature in enumerate(
    features,
    start=1
):

    print(
        f"{number:02d}. {feature}"
    )


# ============================================================
# TARGET DEFINITIONS
# ============================================================

print("\nTarget definitions:")

print("0 = Natural")
print("1 = Industrial")
print("2 = Mining")
print("3 = Other_Uncertain")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "XGBOOST DATASET CREATION COMPLETED SUCCESSFULLY"
)

print("=" * 70)