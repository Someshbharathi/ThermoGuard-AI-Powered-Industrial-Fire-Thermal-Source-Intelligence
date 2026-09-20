import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

from config.paths import (
    FIRMS_EVENTS,
    SATELLITE_CANDIDATES,
)

INPUT_FILE = FIRMS_EVENTS
OUTPUT_FILE = SATELLITE_CANDIDATES

SAMPLES_PER_LABEL = 500
RANDOM_SEED = 42

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("SATELLITE CANDIDATE SELECTION")
print("=" * 65)

print("\nLoading FIRMS event dataset...")

df = pd.read_csv(INPUT)

print(f"Total FIRMS rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "thermal_location_id",
    "latitude",
    "longitude",
    "acq_datetime",
    "frp",
    "confidence",
    "weak_label",
    "label_confidence"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ============================================================
# ONE REPRESENTATIVE ROW PER THERMAL LOCATION
# ============================================================

print("\nCreating one representative row per thermal location...")

# Highest FRP observation represents the location
df = df.sort_values(
    ["thermal_location_id", "frp"],
    ascending=[True, False]
)

locations = df.drop_duplicates(
    subset="thermal_location_id",
    keep="first"
).copy()

print(
    f"Unique thermal locations: "
    f"{len(locations):,}"
)

# ============================================================
# AVAILABLE LABELS
# ============================================================

print("\nAvailable weak labels:")

print(
    locations["weak_label"]
    .value_counts(dropna=False)
    .to_string()
)

# ============================================================
# LABELS WE WANT FOR SATELLITE IMAGERY
# ============================================================

labels_to_use = [
    "Possible_Industrial",
    "Likely_Industrial",
    "Industrial_Persistent",
    "Likely_Mining",
    "Possible_Natural",
    "Unknown"
]

rng = np.random.default_rng(RANDOM_SEED)

selected_parts = []

print("\nSelecting satellite candidates...\n")

for label in labels_to_use:

    subset = locations[
        locations["weak_label"] == label
    ].copy()

    available = len(subset)

    if available == 0:
        print(f"{label:25s} -> NO DATA")
        continue

    sample_size = min(
        SAMPLES_PER_LABEL,
        available
    )

    selected_indices = rng.choice(
        subset.index.to_numpy(),
        size=sample_size,
        replace=False
    )

    selected = subset.loc[selected_indices].copy()

    selected_parts.append(selected)

    print(
        f"{label:25s} "
        f"Available: {available:7,} "
        f"Selected: {sample_size:5,}"
    )

# ============================================================
# COMBINE
# ============================================================

if not selected_parts:
    raise RuntimeError("No candidates were selected.")

candidates = pd.concat(
    selected_parts,
    ignore_index=True
)

# ============================================================
# CREATE SATELLITE ID
# ============================================================

candidates.insert(
    0,
    "satellite_id",
    [
        f"SAT_{i:05d}"
        for i in range(1, len(candidates) + 1)
    ]
)

# ============================================================
# SORT
# ============================================================

candidates = candidates.sort_values(
    ["weak_label", "thermal_location_id"]
).reset_index(drop=True)

# ============================================================
# SAVE
# ============================================================

candidates.to_csv(
    OUTPUT,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SATELLITE CANDIDATE DATASET CREATED")
print("=" * 65)

print(f"\nOutput file: {OUTPUT}")
print(f"Rows: {len(candidates):,}")
print(f"Columns: {len(candidates.columns):,}")

print("\nFinal label distribution:")

print(
    candidates["weak_label"]
    .value_counts()
    .to_string()
)

print(
    f"\nUnique thermal locations: "
    f"{candidates['thermal_location_id'].nunique():,}"
)

print("\nFirst 10 candidates:")

print(
    candidates[
        [
            "satellite_id",
            "thermal_location_id",
            "latitude",
            "longitude",
            "acq_datetime",
            "frp",
            "confidence",
            "weak_label",
            "label_confidence"
        ]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 65)
print("DONE!")
print("=" * 65)