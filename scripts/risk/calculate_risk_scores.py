import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# THERMOGUARD
# RISK SCORING ENGINE
# ============================================================

from config.paths import (
    SATELLITE_CANDIDATES,
    PREDICTIONS_DIR,
    RISK_DIR,
)
FUSION_FILE = PREDICTIONS_DIR / "fusion" / "fusion_probabilities.csv"

CANDIDATES_FILE = SATELLITE_CANDIDATES

OUTPUT_DIR = RISK_DIR
OUTPUT_FILE = OUTPUT_DIR / "thermoguard_risk_scores.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("THERMOGUARD RISK ENGINE")
print("=" * 60)


# ============================================================
# LOAD FUSION RESULTS
# ============================================================

df = pd.read_csv(
    FUSION_FILE
)

print(
    f"Loaded {len(df)} fusion samples."
)


# ============================================================
# LOAD SATELLITE CANDIDATE DATA
# ============================================================

candidates = pd.read_csv(
    CANDIDATE_FILE
)

print(
    f"Loaded {len(candidates)} candidate records."
)


# ============================================================
# REQUIRED FEATURES
# ============================================================

risk_features = [

    # Identification
    "satellite_id",

    # Location
    "latitude",
    "longitude",

    # Weak label
    "weak_label",

    # -------------------------
    # Thermal features
    # -------------------------

    "bright_ti4",
    "bright_ti5",
    "frp",

    # -------------------------
    # OSM spatial features
    # -------------------------

    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",

    "distance_to_petroleum_well_km",

    "distance_to_quarry_km",
    "distance_to_mining_km",

    # -------------------------
    # Persistence features
    # -------------------------

    "detection_count",
    "active_days",
    "duration_days",

    "detections_per_active_day",
    "persistence_ratio",

    "detections_7d",
    "detections_30d",
    "detections_90d",

    # -------------------------
    # Event features
    # -------------------------

    "event_detection_count",
    "event_active_days",
    "event_duration_days",

    "event_mean_frp",
    "event_max_frp",

    "event_satellite_count"
]


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing = [
    column
    for column in risk_features
    if column not in candidates.columns
]

if missing:

    print("\nERROR")
    print("The following columns are missing:")
    print(missing)

    print("\nAvailable columns:")
    print(candidates.columns.tolist())

    raise ValueError(
        f"Missing required columns: {missing}"
    )


print(
    f"All {len(risk_features)} required risk features found."
)


# ============================================================
# SELECT FEATURES
# ============================================================

features = candidates[
    risk_features
].copy()


# ============================================================
# MERGE FUSION + RISK FEATURES
# ============================================================

df = df.merge(
    features,
    on="satellite_id",
    how="left",
    suffixes=("", "_candidate")
)


print(
    f"Rows after merge: {len(df)}"
)


# ============================================================
# CHECK MERGE
# ============================================================

missing_after_merge = df[
    "frp"
].isna().sum()

if missing_after_merge > 0:

    print(
        f"\nWARNING: {missing_after_merge} "
        "rows have missing candidate data."
    )

else:

    print(
        "All fusion samples matched candidate data."
    )


# ============================================================
# HELPER FUNCTION
# ============================================================

def percentile_score(
    series,
    low=5,
    high=95
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    valid_values = values[
        np.isfinite(values)
    ]

    if len(valid_values) == 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    lower = np.percentile(
        valid_values,
        low
    )

    upper = np.percentile(
        valid_values,
        high
    )

    if upper <= lower:

        return pd.Series(
            50.0,
            index=series.index
        )

    score = (
        (values - lower)
        /
        (upper - lower)
    ) * 100

    score = score.clip(
        0,
        100
    )

    return score.fillna(0)


# ============================================================
# 1. THERMAL INTENSITY SCORE
# ============================================================

print("\nCalculating thermal intensity score...")

frp_score = percentile_score(
    df["frp"]
)

brightness_score = percentile_score(
    df["bright_ti4"]
)

thermal_score = (
    0.70 * frp_score
    +
    0.30 * brightness_score
)

thermal_score = thermal_score.clip(
    0,
    100
)


# ============================================================
# 2. PERSISTENCE SCORE
# ============================================================

print(
    "Calculating persistence score..."
)

active_days_score = percentile_score(
    df["active_days"]
)

duration_score = percentile_score(
    df["duration_days"]
)

detection_count_score = percentile_score(
    df["detection_count"]
)

persistence_ratio_score = percentile_score(
    df["persistence_ratio"]
)

persistence_score = (

    0.30 * active_days_score

    +

    0.25 * duration_score

    +

    0.25 * detection_count_score

    +

    0.20 * persistence_ratio_score
)

persistence_score = (
    persistence_score
    .clip(0, 100)
)


# ============================================================
# DISTANCE-BASED SCORE
# ============================================================

def proximity_score(
    series
):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    # Distance where closer = higher score
    reference = values.quantile(
        0.95
    )

    if pd.isna(reference) or reference <= 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    score = (
        1
        -
        (
            values
            /
            reference
        )
    )

    return (
        score
        .clip(0, 1)
        .fillna(0)
        * 100
    )


# ============================================================
# 3. INDUSTRIAL CONTEXT SCORE
# ============================================================

print(
    "Calculating industrial context score..."
)

industrial_area_score = proximity_score(
    df["distance_to_industrial_area_km"]
)

industrial_works_score = proximity_score(
    df["distance_to_industrial_works_km"]
)

storage_tank_score = proximity_score(
    df["distance_to_storage_tank_km"]
)

industrial_score = (

    0.50 * industrial_area_score

    +

    0.30 * industrial_works_score

    +

    0.20 * storage_tank_score
)

industrial_score = (
    industrial_score
    .clip(0, 100)
)


# ============================================================
# 4. MINING CONTEXT SCORE
# ============================================================

print(
    "Calculating mining context score..."
)

quarry_score = proximity_score(
    df["distance_to_quarry_km"]
)

mining_score_component = proximity_score(
    df["distance_to_mining_km"]
)

mining_score = (

    0.55 * quarry_score

    +

    0.45 * mining_score_component
)

mining_score = (
    mining_score
    .clip(0, 100)
)


# ============================================================
# 5. MODEL CONFIDENCE
# ============================================================

print(
    "Calculating model confidence..."
)


# ------------------------------------------------------------
# XGBoost probabilities
# ------------------------------------------------------------

xgb_probability_columns = [

    "xgb_Natural",
    "xgb_Industrial",
    "xgb_Mining",
    "xgb_Other_Uncertain"
]


xgb_columns_available = all(
    column in df.columns
    for column in xgb_probability_columns
)


if xgb_columns_available:

    xgb_probability_matrix = df[
        xgb_probability_columns
    ].values

    xgb_confidence = (
        np.max(
            xgb_probability_matrix,
            axis=1
        )
        * 100
    )

else:

    xgb_confidence = np.zeros(
        len(df)
    )


# ------------------------------------------------------------
# Fusion probabilities
# ------------------------------------------------------------

fusion_probability_columns = [

    "fusion_50_50",
    "fusion_70_30",
    "fusion_30_70"
]


fusion_columns_available = all(
    column in df.columns
    for column in fusion_probability_columns
)


# The fusion CSV currently stores the prediction
# probability for each fusion configuration.
#
# For the risk engine we use the 70/30 configuration
# when available because this configuration matched
# the XGBoost result in the exploratory experiment.

if "fusion_70_30" in df.columns:

    fusion_confidence = (
        pd.to_numeric(
            df["fusion_70_30"],
            errors="coerce"
        )
        .fillna(0)
        * 100
    )

else:

    fusion_confidence = (
        pd.Series(
            xgb_confidence,
            index=df.index
        )
    )


# ------------------------------------------------------------
# Final model confidence
# ------------------------------------------------------------

if xgb_columns_available:

    model_confidence = (
        0.70 * xgb_confidence
        +
        0.30 * fusion_confidence
    )

else:

    model_confidence = fusion_confidence


model_confidence = (
    pd.Series(
        model_confidence,
        index=df.index
    )
    .clip(0, 100)
)


# ============================================================
# 6. FINAL RISK SCORE
# ============================================================

print(
    "Calculating final risk score..."
)


# ------------------------------------------------------------
# Risk weighting
#
# Thermal intensity      30%
# Persistence            25%
# Industrial context     20%
# Mining context         10%
# Model confidence       15%
# ------------------------------------------------------------

risk_score = (

    0.30 * thermal_score

    +

    0.25 * persistence_score

    +

    0.20 * industrial_score

    +

    0.10 * mining_score

    +

    0.15 * model_confidence
)


risk_score = (
    risk_score
    .clip(0, 100)
)


# ============================================================
# 7. RISK LEVEL
# ============================================================

def get_risk_level(
    score
):

    if score < 25:

        return "Low"

    elif score < 50:

        return "Medium"

    elif score < 75:

        return "High"

    else:

        return "Critical"


df["risk_score"] = (
    risk_score
)

df["risk_level"] = (
    df["risk_score"]
    .apply(get_risk_level)
)


# ============================================================
# 8. PREDICTED CLASS
# ============================================================

print(
    "Determining predicted class..."
)


class_names = {

    0: "Natural",

    1: "Industrial",

    2: "Mining",

    3: "Other_Uncertain"
}


# ------------------------------------------------------------
# Determine class from XGBoost probabilities
# ------------------------------------------------------------

if xgb_columns_available:

    predicted_class_id = np.argmax(
        xgb_probability_matrix,
        axis=1
    )

    df["predicted_class"] = [
        class_names[
            int(class_id)
        ]
        for class_id
        in predicted_class_id
    ]

else:

    df["predicted_class"] = (
        "Unknown"
    )


# ============================================================
# 9. EXPLANATION GENERATION
# ============================================================

print(
    "Generating risk explanations..."
)


frp_threshold = df[
    "frp"
].quantile(0.75)

active_days_threshold = df[
    "active_days"
].quantile(0.75)

detections_30_threshold = df[
    "detections_30d"
].quantile(0.75)


def generate_explanation(
    row
):

    reasons = []


    # --------------------------------------------------------
    # Thermal intensity
    # --------------------------------------------------------

    if (
        pd.notna(row["frp"])
        and
        row["frp"] >= frp_threshold
    ):

        reasons.append(
            "high thermal intensity"
        )


    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    if (
        pd.notna(row["active_days"])
        and
        row["active_days"]
        >= active_days_threshold
    ):

        reasons.append(
            "persistent thermal activity"
        )


    # --------------------------------------------------------
    # Industrial area
    # --------------------------------------------------------

    if (
        pd.notna(
            row[
                "distance_to_industrial_area_km"
            ]
        )
        and
        row[
            "distance_to_industrial_area_km"
        ] <= 5
    ):

        reasons.append(
            "near industrial area"
        )


    # --------------------------------------------------------
    # Industrial infrastructure
    # --------------------------------------------------------

    if (
        pd.notna(
            row[
                "distance_to_industrial_works_km"
            ]
        )
        and
        row[
            "distance_to_industrial_works_km"
        ] <= 5
    ):

        reasons.append(
            "near industrial infrastructure"
        )


    # --------------------------------------------------------
    # Storage infrastructure
    # --------------------------------------------------------

    if (
        pd.notna(
            row[
                "distance_to_storage_tank_km"
            ]
        )
        and
        row[
            "distance_to_storage_tank_km"
        ] <= 5
    ):

        reasons.append(
            "near storage infrastructure"
        )


    # --------------------------------------------------------
    # Quarry
    # --------------------------------------------------------

    if (
        pd.notna(
            row[
                "distance_to_quarry_km"
            ]
        )
        and
        row[
            "distance_to_quarry_km"
        ] <= 5
    ):

        reasons.append(
            "near quarry activity"
        )


    # --------------------------------------------------------
    # Mining
    # --------------------------------------------------------

    if (
        pd.notna(
            row[
                "distance_to_mining_km"
            ]
        )
        and
        row[
            "distance_to_mining_km"
        ] <= 5
    ):

        reasons.append(
            "near mining activity"
        )


    # --------------------------------------------------------
    # Repeated detections
    # --------------------------------------------------------

    if (
        pd.notna(
            row["detections_30d"]
        )
        and
        row["detections_30d"]
        >= detections_30_threshold
    ):

        reasons.append(
            "repeated detections over 30 days"
        )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if len(reasons) == 0:

        reasons.append(
            "limited risk indicators detected"
        )


    return "; ".join(
        reasons
    )


df["risk_explanation"] = (
    df.apply(
        generate_explanation,
        axis=1
    )
)


# ============================================================
# 10. COMPONENT SCORES
# ============================================================

df["thermal_score"] = (
    thermal_score
)

df["persistence_score"] = (
    persistence_score
)

df["industrial_score"] = (
    industrial_score
)

df["mining_score"] = (
    mining_score
)

df["model_confidence"] = (
    model_confidence
)


# ============================================================
# 11. OUTPUT COLUMNS
# ============================================================

output_columns = [

    # Identification
    "satellite_id",

    "thermal_location_id",

    # Location
    "latitude",
    "longitude",

    # Date
    "acq_date",

    # Labels
    "weak_label",

    "predicted_class",

    # Model probabilities
    "xgb_Natural",
    "xgb_Industrial",
    "xgb_Mining",
    "xgb_Other_Uncertain",

    # Fusion
    "fusion_50_50",
    "fusion_70_30",
    "fusion_30_70",

    # Thermal
    "frp",
    "bright_ti4",
    "bright_ti5",

    # Persistence
    "active_days",
    "duration_days",
    "detection_count",

    "detections_7d",
    "detections_30d",
    "detections_90d",

    # Spatial
    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",

    "distance_to_quarry_km",
    "distance_to_mining_km",

    # Component scores
    "thermal_score",
    "persistence_score",
    "industrial_score",
    "mining_score",
    "model_confidence",

    # Final risk
    "risk_score",
    "risk_level",

    # Explanation
    "risk_explanation"
]


# Keep only columns actually available

output_columns = [
    column
    for column in output_columns
    if column in df.columns
]


result = df[
    output_columns
].copy()


# ============================================================
# ROUND NUMERIC VALUES
# ============================================================

numeric_columns = result.select_dtypes(
    include=["float64", "float32"]
).columns

result[
    numeric_columns
] = result[
    numeric_columns
].round(4)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("RISK SCORING COMPLETE")
print("=" * 60)


print(
    f"\nSamples processed: {len(result)}"
)


print(
    "\nRisk distribution:"
)

print(
    result[
        "risk_level"
    ]
    .value_counts()
    .reindex(
        [
            "Low",
            "Medium",
            "High",
            "Critical"
        ],
        fill_value=0
    )
)


print(
    "\nAverage risk score:",
    round(
        result[
            "risk_score"
        ].mean(),
        2
    )
)


print(
    "\nMinimum risk score:",
    round(
        result[
            "risk_score"
        ].min(),
        2
    )
)


print(
    "Maximum risk score:",
    round(
        result[
            "risk_score"
        ].max(),
        2
    )
)


print(
    "\nPredicted classes:"
)

print(
    result[
        "predicted_class"
    ]
    .value_counts()
)


print(
    "\nTop 10 highest-risk events:"
)

top_events = result.sort_values(
    "risk_score",
    ascending=False
).head(10)

print(
    top_events[
        [
            "satellite_id",
            "predicted_class",
            "risk_score",
            "risk_level",
            "frp",
            "active_days",
            "risk_explanation"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nOutput file:"
)

print(
    OUTPUT_FILE
)


print(
    "\nDone."
)