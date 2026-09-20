import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    XGBOOST_DATA_DIR,
    RESULTS_DIR,
)
TRAIN_FILE = XGBOOST_DATA_DIR / "train.csv"
VALIDATION_FILE = XGBOOST_DATA_DIR / "validation.csv"
TEST_FILE = XGBOOST_DATA_DIR / "test.csv"

OUTPUT_DIR = RESULTS_DIR / "metrics" / "xgboost_ablation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASS MAPPING
# ============================================================

CLASS_NAMES = [
    "Natural",
    "Industrial",
    "Mining",
    "Other_Uncertain"
]

CLASS_MAP = {
    "Natural": 0,
    "Industrial": 1,
    "Mining": 2,
    "Other_Uncertain": 3
}


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("THERMOGUARD XGBOOST ABLATION STUDY")
print("=" * 70)

print("\nLoading datasets...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "validation.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")


# ============================================================
# ALL FEATURES
# ============================================================

ALL_FEATURES = [
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
# FEATURE GROUPS
# ============================================================

OSM_PROXIMITY = [
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
]

OSM_DISTANCE = [
    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km",
]

PERSISTENCE = [
    "detection_count",
    "active_days",
    "duration_days",
    "detections_per_active_day",
    "persistence_ratio",
    "detections_7d",
    "detections_30d",
    "detections_90d",
]

EVENT = [
    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",
]

TEMPORAL = [
    "month",
    "day_of_year",
    "hour",
]


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

experiments = {

    "A_full_baseline": {
        "remove": [],
        "description": "All 53 features"
    },

    "B_no_osm_proximity": {
        "remove": OSM_PROXIMITY,
        "description": "Remove all threshold-based OSM proximity flags"
    },

    "C_no_osm_persistence_event": {
        "remove": OSM_PROXIMITY + OSM_DISTANCE + PERSISTENCE + EVENT,
        "description": "Remove OSM, persistence and event-derived features"
    },

    "D_firms_only": {
        "remove": OSM_PROXIMITY + OSM_DISTANCE + PERSISTENCE + EVENT + TEMPORAL,
        "description": "Use only raw FIRMS thermal/signal features"
    }
}


# ============================================================
# MODEL FUNCTION
# ============================================================

def run_experiment(name, config):

    print("\n")
    print("=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    remove_features = config["remove"]

    features = [
        f for f in ALL_FEATURES
        if f not in remove_features
    ]

    print(f"\nDescription: {config['description']}")
    print(f"Features used: {len(features)}")

    print("\nFeatures:")
    for f in features:
        print("  -", f)

    # --------------------------------------------------------
    # Prepare X/y
    # --------------------------------------------------------

    X_train = train_df[features].copy()
    X_val = val_df[features].copy()
    X_test = test_df[features].copy()

    # Encode target classes
    y_train = train_df["target"].map(CLASS_MAP).values
    y_val = val_df["target"].map(CLASS_MAP).values
    y_test = test_df["target"].map(CLASS_MAP).values
    if np.any(pd.isna(y_train)) or np.any(pd.isna(y_val)) or np.any(pd.isna(y_test)):
    	raise ValueError("Unknown target class found in target column")

    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    y_test = y_test.astype(int)
    # --------------------------------------------------------
    # Class-balanced weights
    # --------------------------------------------------------

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=4,

        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        min_child_weight=2,
        gamma=0,

        reg_alpha=0,
        reg_lambda=1,

        eval_metric="mlogloss",

        tree_method="hist",

        random_state=42,
        n_jobs=-1
    )

    print("\nTraining...")

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val)
        ],
        verbose=False
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    def calculate_metrics(y_true, y_pred):

        return {
            "accuracy": float(
                accuracy_score(y_true, y_pred)
            ),

            "precision_weighted": float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

            "recall_weighted": float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

            "f1_weighted": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

            "f1_macro": float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            )
        }

    train_metrics = calculate_metrics(y_train, train_pred)
    val_metrics = calculate_metrics(y_val, val_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    # --------------------------------------------------------
    # Print metrics
    # --------------------------------------------------------

    print("\nTRAIN")
    print(train_metrics)

    print("\nVALIDATION")
    print(val_metrics)

    print("\nTEST")
    print(test_metrics)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        test_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print("\nTEST CLASSIFICATION REPORT")
    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred
    )

    print("\nCONFUSION MATRIX")
    print(cm)

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    exp_dir = os.path.join(
        OUTPUT_DIR,
        name
    )

    os.makedirs(exp_dir, exist_ok=True)

    model.save_model(
        os.path.join(
            exp_dir,
            "model.json"
        )
    )

    metrics = {
        "experiment": name,
        "description": config["description"],
        "feature_count": len(features),
        "features": features,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
        "confusion_matrix": cm.tolist()
    }

    with open(
        os.path.join(exp_dir, "metrics.json"),
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    importance_df.to_csv(
        os.path.join(
            exp_dir,
            "feature_importance.csv"
        ),
        index=False
    )

    predictions = test_df[
    [
        "satellite_id",
        "thermal_location_id",
        "resnet_label"
    ]
].copy()

    predictions["actual"] = y_test
    predictions["predicted"] = test_pred

    predictions.to_csv(
        os.path.join(
            exp_dir,
            "test_predictions.csv"
        ),
        index=False
    )

    print(
        f"\nSaved experiment to:\n{exp_dir}"
    )

    return {
        "experiment": name,
        "features": len(features),
        "train_accuracy": train_metrics["accuracy"],
        "validation_accuracy": val_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_weighted"],
        "test_macro_f1": test_metrics["f1_macro"]
    }


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================

results = []

for name, config in experiments.items():

    result = run_experiment(
        name,
        config
    )

    results.append(result)


# ============================================================
# SUMMARY
# ============================================================

summary_df = pd.DataFrame(results)

summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ablation_summary.csv"
    ),
    index=False
)

print("\n")
print("=" * 70)
print("ABLATION STUDY SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)