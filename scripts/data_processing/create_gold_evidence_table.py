import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    SATELLITE_CANDIDATES,
    VERIFICATION_DATA_DIR,
)
GOLD_TEST_FILE = VERIFICATION_DATA_DIR / "gold_test_set" / "gold_test_173.csv"

CANDIDATES_FILE = SATELLITE_CANDIDATES

OUTPUT_FILE = VERIFICATION_DATA_DIR / "gold_test_set" / "gold_evidence_173.csv"


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)




# ============================================================
# LOAD FILES
# ============================================================

print("Loading gold test set...")

gold = pd.read_csv(GOLD_TEST)

print(f"Gold test rows: {len(gold)}")

print("\nLoading satellite candidates...")

candidates = pd.read_csv(CANDIDATES)

print(f"Candidate rows: {len(candidates)}")


# ============================================================
# CHECK REQUIRED IDENTIFIER
# ============================================================

if "satellite_id" not in gold.columns:
    raise ValueError(
        "satellite_id missing from gold_test_173.csv"
    )

if "satellite_id" not in candidates.columns:
    raise ValueError(
        "satellite_id missing from satellite_candidates.csv"
    )


# ============================================================
# REMOVE DUPLICATES FROM CANDIDATES
# ============================================================

duplicate_candidates = candidates["satellite_id"].duplicated().sum()

print(
    f"\nDuplicate candidate satellite IDs: "
    f"{duplicate_candidates}"
)

if duplicate_candidates > 0:
    raise ValueError(
        "Duplicate satellite IDs found in satellite_candidates.csv"
    )


# ============================================================
# EVIDENCE FEATURES
# ============================================================

evidence_columns = [

    # --------------------------------------------------------
    # Thermal / FIRMS
    # --------------------------------------------------------

    "bright_ti4",
    "bright_ti5",
    "frp",
    "confidence",
    "confidence_score",

    # --------------------------------------------------------
    # Spatial / OSM
    # --------------------------------------------------------

    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km",

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

    # --------------------------------------------------------
    # Persistence
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
    # Event-level evidence
    # --------------------------------------------------------

    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",

    # --------------------------------------------------------
    # Original metadata
    # --------------------------------------------------------

    "month",
    "day_of_year",
    "hour",

    "weak_label",
    "label_confidence"
]


# ============================================================
# CHECK WHICH COLUMNS EXIST
# ============================================================

available = []

missing = []

for column in evidence_columns:

    if column in candidates.columns:
        available.append(column)
    else:
        missing.append(column)


print("\nEvidence columns found:", len(available))

if missing:

    print("\nColumns not available:")
    for column in missing:
        print("  -", column)


# ============================================================
# SELECT CANDIDATE EVIDENCE
# ============================================================

candidate_evidence = candidates[
    ["satellite_id"] + available
].copy()


# ============================================================
# MERGE WITH GOLD TEST
# ============================================================

print("\nMerging evidence...")

evidence = gold.merge(
    candidate_evidence,
    on="satellite_id",
    how="left",
    suffixes=("", "_candidate")
)


# ============================================================
# CHECK MATCHING
# ============================================================

print(
    "\nRows after merge:",
    len(evidence)
)

missing_matches = evidence[
    evidence["bright_ti4"].isna()
]["satellite_id"]

print(
    "Samples without candidate evidence:",
    len(missing_matches)
)

if len(missing_matches) > 0:

    print("\nMissing IDs:")

    for sid in missing_matches:
        print("  -", sid)


# ============================================================
# REMOVE DUPLICATE COLUMNS
# ============================================================

# Keep the original gold-test metadata where possible.
# Candidate columns that duplicate existing columns are removed.

duplicate_metadata = [
    "weak_label",
    "label_confidence"
]

for column in duplicate_metadata:

    candidate_column = f"{column}_candidate"

    if candidate_column in evidence.columns:

        evidence.drop(
            columns=[candidate_column],
            inplace=True
        )


# ============================================================
# ADD VERIFICATION GUIDANCE COLUMNS
# ============================================================

# These remain BLANK intentionally.
# They will be filled during independent verification.

evidence["visual_review_label"] = ""

evidence["verification_confidence"] = ""

evidence["verification_basis"] = ""

evidence["reviewer_notes"] = ""


# ============================================================
# SAVE
# ============================================================

evidence.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("GOLD EVIDENCE TABLE CREATED")
print("==========================================")

print(
    f"Rows: {len(evidence)}"
)

print(
    f"Columns: {len(evidence.columns)}"
)

print(
    f"Output:\n{OUTPUT_FILE}"
)

print("\nWeak-label distribution:")

print(
    evidence["weak_label"].value_counts()
)

print("\nVerification columns:")

print(
    [
        "visual_review_label",
        "verification_confidence",
        "verification_basis",
        "reviewer_notes"
    ]
)

print("\nDone.")