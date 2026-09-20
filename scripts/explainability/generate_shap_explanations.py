import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================
from config.paths import (
    XGBOOST_DATA_DIR,
    XGBOOST_MODEL_DIR,
    SHAP_DIR,
)
TEST_FILE = XGBOOST_DATA_DIR / "test.csv"

MODEL_FILE = XGBOOST_MODEL_DIR / "thermoguard_xgboost.json"

OUTPUT_DIR = SHAP_DIR
PLOTS_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# XGBOOST FEATURES
# ============================================================

FEATURES = [

    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "confidence_score",

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

    "detection_count",
    "active_days",
    "duration_days",
    "detections_per_active_day",
    "persistence_ratio",

    "detections_7d",
    "detections_30d",
    "detections_90d",

    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",

    "month",
    "day_of_year",
    "hour"
]


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Natural",
    1: "Industrial",
    2: "Mining",
    3: "Other_Uncertain"
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("THERMOGUARD SHAP EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"Test rows: {len(test_df)}"
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in test_df.columns
]

if missing_features:

    raise ValueError(
        f"Missing XGBoost features: {missing_features}"
    )


X_test = test_df[
    FEATURES
].copy()


# ============================================================
# CHECK NUMERIC DATA
# ============================================================

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)

if X_test.isna().sum().sum() > 0:

    print(
        "\nWARNING: Missing feature values found."
    )

    print(
        X_test.isna()
        .sum()
        .loc[
            lambda x: x > 0
        ]
    )

    X_test = X_test.fillna(
        X_test.median()
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained XGBoost model...")

model = xgb.XGBClassifier()

model.load_model(
    MODEL_FILE
)

print(
    "XGBoost model loaded successfully."
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)

predicted_classes = [
    CLASS_NAMES[int(x)]
    for x in predictions
]

prediction_confidence = (
    np.max(
        probabilities,
        axis=1
    )
)


print(
    "Predictions generated."
)


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(
    model
)


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print(
    "Calculating SHAP values..."
)

shap_values = explainer.shap_values(
    X_test
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

# Depending on the SHAP/XGBoost version,
# multiclass output may be:
#
#   (samples, features, classes)
#
# or:
#
#   list of arrays
#
# We convert everything into:
#
#   (samples, features, classes)


if isinstance(
    shap_values,
    list
):

    shap_array = np.stack(
        shap_values,
        axis=2
    )

else:

    shap_array = np.asarray(
        shap_values
    )


print(
    "SHAP array shape:",
    shap_array.shape
)


# ============================================================
# VALIDATE SHAPE
# ============================================================

if shap_array.ndim != 3:

    raise ValueError(
        "Unexpected SHAP output shape: "
        f"{shap_array.shape}"
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

print(
    "\nCalculating global feature importance..."
)


# Mean absolute SHAP value across:
# samples and classes

global_importance = (
    np.abs(
        shap_array
    )
    .mean(axis=(0, 2))
)


global_importance_df = pd.DataFrame({

    "feature": FEATURES,

    "mean_abs_shap": global_importance

})


global_importance_df = (
    global_importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


global_importance_df[
    "rank"
] = (
    np.arange(
        1,
        len(global_importance_df) + 1
    )
)


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

global_file = (
    OUTPUT_DIR
    / "global_feature_importance.csv"
)

global_importance_df.to_csv(
    global_file,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print(
    "\nTop 20 SHAP features:"
)

print(
    global_importance_df
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# EVENT-LEVEL SHAP EXPLANATIONS
# ============================================================

print(
    "\nGenerating event-level explanations..."
)


event_rows = []


for i in range(
    len(X_test)
):

    predicted_class_id = int(
        predictions[i]
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_class_id
        ]
    )

    # SHAP values for the predicted class
    values = shap_array[
        i,
        :,
        predicted_class_id
    ]

    feature_values = (
        X_test.iloc[i]
        .values
    )

    # Sort by absolute contribution
    order = np.argsort(
        np.abs(values)
    )[::-1]


    top_positive = []

    top_negative = []


    for idx in order:

        feature = FEATURES[
            idx
        ]

        shap_value = float(
            values[idx]
        )

        actual_value = float(
            feature_values[idx]
        )

        record = (
            feature,
            shap_value,
            actual_value
        )


        if shap_value > 0:

            if len(top_positive) < 5:

                top_positive.append(
                    record
                )

        elif shap_value < 0:

            if len(top_negative) < 5:

                top_negative.append(
                    record
                )


        if (
            len(top_positive) >= 5
            and
            len(top_negative) >= 5
        ):

            break


    # --------------------------------------------------------
    # Format explanation
    # --------------------------------------------------------

    positive_text = []

    for (
        feature,
        value,
        actual
    ) in top_positive:

        positive_text.append(
            f"{feature} (+{value:.4f}, value={actual:.4f})"
        )


    negative_text = []

    for (
        feature,
        value,
        actual
    ) in top_negative:

        negative_text.append(
            f"{feature} ({value:.4f}, value={actual:.4f})"
        )


    # --------------------------------------------------------
    # Identify sample
    # --------------------------------------------------------

    if "satellite_id" in test_df.columns:

        satellite_id = (
            test_df.iloc[i][
                "satellite_id"
            ]
        )

    else:

        satellite_id = i


    # --------------------------------------------------------
    # Create row
    # --------------------------------------------------------

    row = {

        "satellite_id":
            satellite_id,

        "predicted_class":
            predicted_class,

        "prediction_confidence":
            float(
                prediction_confidence[i]
            ),

        "top_positive_features":
            " | ".join(
                positive_text
            ),

        "top_negative_features":
            " | ".join(
                negative_text
            )
    }


    # Add top 10 individual SHAP values

    for rank, idx in enumerate(
        order[:10],
        start=1
    ):

        row[
            f"feature_{rank}"
        ] = FEATURES[idx]

        row[
            f"shap_value_{rank}"
        ] = float(
            values[idx]
        )

        row[
            f"feature_value_{rank}"
        ] = float(
            feature_values[idx]
        )


    event_rows.append(
        row
    )


event_explanations = pd.DataFrame(
    event_rows
)


# ============================================================
# SAVE EVENT EXPLANATIONS
# ============================================================

event_file = (
    OUTPUT_DIR
    / "event_explanations.csv"
)

event_explanations.to_csv(
    event_file,
    index=False
)


# ============================================================
# SAVE RAW SHAP VALUES
# ============================================================

print(
    "\nSaving SHAP values..."
)


raw_rows = []


for i in range(
    len(X_test)
):

    satellite_id = (
        test_df.iloc[i][
            "satellite_id"
        ]
        if "satellite_id"
        in test_df.columns
        else i
    )

    predicted_class_id = int(
        predictions[i]
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_class_id
        ]
    )


    for feature_index, feature in enumerate(
        FEATURES
    ):

        for class_index in range(4):

            raw_rows.append({

                "satellite_id":
                    satellite_id,

                "predicted_class":
                    predicted_class,

                "shap_class":
                    CLASS_NAMES[
                        class_index
                    ],

                "feature":
                    feature,

                "feature_value":
                    float(
                        X_test.iloc[i][
                            feature
                        ]
                    ),

                "shap_value":
                    float(
                        shap_array[
                            i,
                            feature_index,
                            class_index
                        ]
                    )
            })


raw_shap_df = pd.DataFrame(
    raw_rows
)


raw_shap_file = (
    OUTPUT_DIR
    / "shap_values.csv"
)

raw_shap_df.to_csv(
    raw_shap_file,
    index=False
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print(
    "\nCreating SHAP summary plot..."
)


# For multiclass, calculate mean absolute
# SHAP across classes for visualization.

mean_abs_shap_per_feature = (
    np.abs(
        shap_array
    )
    .mean(axis=2)
)


summary_values = (
    shap_array.mean(axis=2)
)


plt.figure(
    figsize=(12, 8)
)


shap.summary_plot(
    summary_values,
    X_test,
    show=False,
    max_display=20
)


plt.tight_layout()


summary_plot = (
    PLOT_DIR
    / "shap_summary.png"
)


plt.savefig(
    summary_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# GLOBAL BAR PLOT
# ============================================================

print(
    "Creating SHAP feature importance plot..."
)


top_n = 20

plot_df = (
    global_importance_df
    .head(top_n)
    .sort_values(
        "mean_abs_shap"
    )
)


plt.figure(
    figsize=(10, 8)
)


plt.barh(
    plot_df[
        "feature"
    ],
    plot_df[
        "mean_abs_shap"
    ]
)


plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "ThermoGuard Global SHAP Feature Importance"
)


plt.tight_layout()


bar_plot = (
    PLOT_DIR
    / "shap_bar.png"
)


plt.savefig(
    bar_plot,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE SUMMARY JSON
# ============================================================

summary = {

    "samples_explained":
        int(len(X_test)),

    "features":
        int(len(FEATURES)),

    "classes":
        CLASS_NAMES,

    "top_features":
        global_importance_df
        .head(20)
        .to_dict(
            orient="records"
        ),

    "files": {

        "global_importance":
            str(global_file),

        "event_explanations":
            str(event_file),

        "raw_shap_values":
            str(raw_shap_file),

        "summary_plot":
            str(summary_plot),

        "bar_plot":
            str(bar_plot)
    }
}


summary_file = (
    OUTPUT_DIR
    / "shap_summary.json"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("SHAP EXPLAINABILITY COMPLETE")
print("=" * 70)


print(
    f"\nSamples explained: {len(X_test)}"
)

print(
    f"Features explained: {len(FEATURES)}"
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nGenerated files:"
)

print(
    global_file
)

print(
    event_file
)

print(
    raw_shap_file
)

print(
    summary_plot
)

print(
    bar_plot
)

print(
    summary_file
)


print(
    "\nDone."
)