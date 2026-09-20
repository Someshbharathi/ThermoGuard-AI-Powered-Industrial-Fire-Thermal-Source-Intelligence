import pandas as pd
import numpy as np

from config.paths import (
    FIRMS_SPATIAL,
    FIRMS_PERSISTENCE,
)

INPUT_FILE = FIRMS_SPATIAL
OUTPUT_FILE = FIRMS_PERSISTENCE

GRID_SIZE = 0.01

print("==========================================")
print("LOADING FIRMS DATA")
print("==========================================")

df = pd.read_csv(INPUT_FILE)

print("Rows loaded:", len(df))

# ------------------------------------------------------------
# DATETIME
# ------------------------------------------------------------

df["acq_datetime"] = pd.to_datetime(
    df["acq_datetime"],
    errors="coerce"
)

df["acq_date"] = pd.to_datetime(
    df["acq_date"],
    errors="coerce"
).dt.date

# ------------------------------------------------------------
# CREATE ~1 KM SPATIAL GRID
# ------------------------------------------------------------

print("\nCreating spatial grid...")

df["grid_lat"] = (
    np.floor(df["latitude"] / GRID_SIZE) * GRID_SIZE
).round(4)

df["grid_lon"] = (
    np.floor(df["longitude"] / GRID_SIZE) * GRID_SIZE
).round(4)

df["thermal_location_id"] = (
    df["grid_lat"].astype(str)
    + "_"
    + df["grid_lon"].astype(str)
)

# ------------------------------------------------------------
# SORT
# ------------------------------------------------------------

print("Sorting detections...")

df = df.sort_values(
    ["thermal_location_id", "acq_datetime"]
).reset_index(drop=True)

# ------------------------------------------------------------
# BASIC PERSISTENCE FEATURES
# ------------------------------------------------------------

print("\nCalculating basic persistence features...")

group = df.groupby(
    "thermal_location_id",
    sort=False
)

df["detection_count"] = group[
    "thermal_location_id"
].transform("size")

df["active_days"] = group[
    "acq_date"
].transform("nunique")

df["first_detection"] = group[
    "acq_datetime"
].transform("min")

df["last_detection"] = group[
    "acq_datetime"
].transform("max")

df["duration_days"] = (
    (
        df["last_detection"]
        - df["first_detection"]
    ).dt.total_seconds() / 86400
)

df["detections_per_active_day"] = (
    df["detection_count"]
    / df["active_days"].replace(0, np.nan)
)

df["persistence_ratio"] = (
    df["active_days"] / 366.0
)

# ------------------------------------------------------------
# FAST TEMPORAL WINDOW COUNTS
# ------------------------------------------------------------

print("\nCalculating fast temporal windows...")

# Arrays for output
n = len(df)

detections_7d = np.zeros(n, dtype=np.int32)
detections_30d = np.zeros(n, dtype=np.int32)
detections_90d = np.zeros(n, dtype=np.int32)

# Convert datetime to int64 nanoseconds
time_values = df["acq_datetime"].astype("int64").to_numpy()

# Group indices
groups = df.groupby(
    "thermal_location_id",
    sort=False
).groups

total_groups = len(groups)

print("Thermal locations:", total_groups)

processed = 0

for location_id, indices in groups.items():

    idx = np.asarray(indices)

    times = time_values[idx]

    # 7 days
    left7 = np.searchsorted(
        times,
        times - 7 * 24 * 60 * 60 * 1_000_000_000,
        side="left"
    )

    detections_7d[idx] = (
        np.arange(len(times)) - left7 + 1
    )

    # 30 days
    left30 = np.searchsorted(
        times,
        times - 30 * 24 * 60 * 60 * 1_000_000_000,
        side="left"
    )

    detections_30d[idx] = (
        np.arange(len(times)) - left30 + 1
    )

    # 90 days
    left90 = np.searchsorted(
        times,
        times - 90 * 24 * 60 * 60 * 1_000_000_000,
        side="left"
    )

    detections_90d[idx] = (
        np.arange(len(times)) - left90 + 1
    )

    processed += 1

    if processed % 10000 == 0:
        print(
            f"Processed locations: "
            f"{processed}/{total_groups}"
        )

# Add results

df["detections_7d"] = detections_7d
df["detections_30d"] = detections_30d
df["detections_90d"] = detections_90d

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print("\n==========================================")
print("SAVING DATASET")
print("==========================================")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n==========================================")
print("COMPLETE")
print("==========================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nOutput:")
print(OUTPUT_FILE)

print("\nPersistence statistics:")

print(
    df[
        [
            "detection_count",
            "active_days",
            "duration_days",
            "detections_per_active_day",
            "persistence_ratio",
            "detections_7d",
            "detections_30d",
            "detections_90d"
        ]
    ].describe()
)