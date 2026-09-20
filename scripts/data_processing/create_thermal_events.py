import pandas as pd
import numpy as np

# ============================================================
# FILES
# ============================================================

from config.paths import (
    FIRMS_PERSISTENCE,
    FIRMS_EVENTS,
)
INPUT_FILE = FIRMS_PERSISTENCE
OUTPUT_FILE = FIRMS_EVENTS


# ============================================================
# SETTINGS
# ============================================================

# Spatial grid used previously
GRID_SIZE = 0.01


# ============================================================
# LOAD DATA
# ============================================================

print("==========================================")
print("LOADING DATA")
print("==========================================")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))


# ============================================================
# DATETIME
# ============================================================

df["acq_datetime"] = pd.to_datetime(
    df["acq_datetime"],
    errors="coerce"
)


# ============================================================
# CREATE EVENT LOCATION
# ============================================================

print("\nCreating thermal event locations...")

df["event_lat"] = (
    np.floor(df["latitude"] / GRID_SIZE)
    * GRID_SIZE
).round(4)

df["event_lon"] = (
    np.floor(df["longitude"] / GRID_SIZE)
    * GRID_SIZE
).round(4)


# ============================================================
# EVENT ID
# ============================================================

df["event_id"] = (
    df["event_lat"].astype(str)
    + "_"
    + df["event_lon"].astype(str)
)


# ============================================================
# BASIC EVENT STATISTICS
# ============================================================

print("Calculating event statistics...")

group = df.groupby(
    "event_id",
    sort=False
)


# Total detections for event

df["event_detection_count"] = group[
    "event_id"
].transform("size")


# Active days

df["event_active_days"] = group[
    "acq_date"
].transform("nunique")


# First observation

df["event_first_detection"] = group[
    "acq_datetime"
].transform("min")


# Last observation

df["event_last_detection"] = group[
    "acq_datetime"
].transform("max")


# Event duration

df["event_duration_days"] = (
    (
        df["event_last_detection"]
        - df["event_first_detection"]
    ).dt.total_seconds()
    / 86400
)


# ============================================================
# AVERAGE / MAX FRP
# ============================================================

df["event_mean_frp"] = group[
    "frp"
].transform("mean")

df["event_max_frp"] = group[
    "frp"
].transform("max")


# ============================================================
# SATELLITE CONFIRMATION
# ============================================================

df["event_satellite_count"] = group[
    "source_satellite"
].transform("nunique")


# ============================================================
# INDUSTRIAL PROXIMITY
# ============================================================

print("Calculating industrial proximity indicators...")


# Minimum distance values are already present
# in the spatially enriched dataset.

industrial_distance_columns = [
    "distance_to_industrial_area_km",
    "distance_to_industrial_km",
    "distance_to_industrial_works_km",
    "distance_to_power_plant_km",
    "distance_to_substation_km",
    "distance_to_refinery_km",
    "distance_to_oil_gas_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km"
]


# ============================================================
# STRONG INDUSTRIAL SIGNALS
# ============================================================

df["strong_industrial_signal"] = (
    (
        df["distance_to_refinery_km"] <= 2
    )
    |
    (
        df["distance_to_oil_gas_km"] <= 2
    )
    |
    (
        df["distance_to_storage_tank_km"] <= 1
    )
    |
    (
        df["distance_to_power_plant_km"] <= 1
    )
    |
    (
        df["distance_to_industrial_works_km"] <= 1
    )
).astype("int8")


# ============================================================
# INDUSTRIAL AREA SIGNAL
# ============================================================

df["industrial_area_signal"] = (
    df["distance_to_industrial_area_km"] <= 2
).astype("int8")


# ============================================================
# PERSISTENCE SIGNAL
# ============================================================

df["persistent_signal"] = (
    (
        df["event_active_days"] >= 10
    )
    |
    (
        df["event_detection_count"] >= 20
    )
).astype("int8")


# ============================================================
# HIGH PERSISTENCE SIGNAL
# ============================================================

df["high_persistence_signal"] = (
    (
        df["event_active_days"] >= 30
    )
    |
    (
        df["event_detection_count"] >= 50
    )
).astype("int8")


# ============================================================
# MULTI-SATELLITE SIGNAL
# ============================================================

df["multi_satellite_signal"] = (
    df["event_satellite_count"] >= 2
).astype("int8")


# ============================================================
# INITIAL WEAK LABEL
# ============================================================

print("\nCreating initial weak labels...")


def assign_label(row):

    strong_industrial = (
        row["strong_industrial_signal"] == 1
    )

    industrial_area = (
        row["industrial_area_signal"] == 1
    )

    persistent = (
        row["persistent_signal"] == 1
    )

    high_persistence = (
        row["high_persistence_signal"] == 1
    )

    refinery = (
        row["distance_to_refinery_km"] <= 2
    )

    oil_gas = (
        row["distance_to_oil_gas_km"] <= 2
    )

    storage = (
        row["distance_to_storage_tank_km"] <= 1
    )

    power = (
        row["distance_to_power_plant_km"] <= 1
    )

    mining = (
        row["distance_to_mining_km"] <= 1
        or
        row["distance_to_quarry_km"] <= 1
    )


    # --------------------------------------------------------
    # INDUSTRIAL / PERSISTENT
    # --------------------------------------------------------

    if (
        strong_industrial
        and high_persistence
    ):
        return "Industrial_Persistent"


    # --------------------------------------------------------
    # LIKELY INDUSTRIAL FIRE
    # --------------------------------------------------------

    if (
        strong_industrial
        and persistent
    ):
        return "Likely_Industrial"


    # --------------------------------------------------------
    # LIKELY GAS / FLARE
    # --------------------------------------------------------

    if (
        (refinery or oil_gas or storage)
        and persistent
    ):
        return "Likely_Gas_Industrial"


    # --------------------------------------------------------
    # LIKELY MINING
    # --------------------------------------------------------

    if (
        mining
        and persistent
    ):
        return "Likely_Mining"


    # --------------------------------------------------------
    # POSSIBLE INDUSTRIAL
    # --------------------------------------------------------

    if (
        industrial_area
        and persistent
    ):
        return "Possible_Industrial"


    # --------------------------------------------------------
    # NATURAL / OTHER
    # --------------------------------------------------------

    if (
        not industrial_area
        and not strong_industrial
        and row["event_active_days"] <= 2
    ):
        return "Possible_Natural"


    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return "Unknown"


df["weak_label"] = df.apply(
    assign_label,
    axis=1
)


# ============================================================
# LABEL CONFIDENCE
# ============================================================

def label_confidence(row):

    score = 0

    if row["strong_industrial_signal"] == 1:
        score += 2

    if row["persistent_signal"] == 1:
        score += 2

    if row["high_persistence_signal"] == 1:
        score += 1

    if row["multi_satellite_signal"] == 1:
        score += 1

    if row["industrial_area_signal"] == 1:
        score += 1

    if score >= 5:
        return "High"

    elif score >= 3:
        return "Medium"

    else:
        return "Low"


df["label_confidence"] = df.apply(
    label_confidence,
    axis=1
)


# ============================================================
# LABEL SUMMARY
# ============================================================

print("\n==========================================")
print("LABEL SUMMARY")
print("==========================================")

print(
    df["weak_label"].value_counts()
)


print("\nConfidence:")

print(
    df["label_confidence"].value_counts()
)


# ============================================================
# SAVE
# ============================================================

print("\n==========================================")
print("SAVING")
print("==========================================")

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("COMPLETE")
print("==========================================")

print("Rows:", len(df))

print("Columns:", len(df.columns))

print("\nOutput:")
print(OUTPUT_FILE)