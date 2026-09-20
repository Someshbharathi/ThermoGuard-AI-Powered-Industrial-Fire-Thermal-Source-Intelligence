import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb

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
# THERMOGUARD
# XGBOOST BASELINE TRAINING
# ============================================================

print("=" * 70)
print("THERMOGUARD - XGBOOST BASELINE TRAINING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    XGBOOST_DATA_DIR,
    XGBOOST_MODEL_DIR,
)
TRAIN_FILE = XGBOOST_DATA_DIR / "train.csv"
VALIDATION_FILE = XGBOOST_DATA_DIR / "validation.csv"
TEST_FILE = XGBOOST_DATA_DIR / "test.csv"

OUTPUT_DIR = XGBOOST_MODEL_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)




# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading datasets...")

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


# ============================================================
# FEATURES
# ============================================================

features = [

    # --------------------------------------------------------
    # THERMAL
    # --------------------------------------------------------

    "bright_ti4",
    "bright_ti5",
    "frp",
    "scan",
    "track",
    "confidence_score",


    # --------------------------------------------------------
    # SPATIAL DISTANCE
    # --------------------------------------------------------

    "distance_to_industrial_area_km",
    "distance_to_industrial_works_km",
    "distance_to_storage_tank_km",
    "distance_to_petroleum_well_km",
    "distance_to_quarry_km",
    "distance_to_mining_km",


    # --------------------------------------------------------
    # SPATIAL PROXIMITY
    # --------------------------------------------------------

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
    # PERSISTENCE
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
    # EVENT
    # --------------------------------------------------------

    "event_detection_count",
    "event_active_days",
    "event_duration_days",
    "event_mean_frp",
    "event_max_frp",
    "event_satellite_count",


    # --------------------------------------------------------
    # TEMPORAL
    # --------------------------------------------------------

    "month",
    "day_of_year",
    "hour"
]


print("\nNumber of features:", len(features))


# ============================================================
# PREPARE X AND Y
# ============================================================

X_train = train[features]
X_validation = validation[features]
X_test = test[features]


# ------------------------------------------------------------
# Convert target labels to numeric class IDs
# ------------------------------------------------------------

label_mapping = {
    "Natural": 0,
    "Industrial": 1,
    "Mining": 2,
    "Other_Uncertain": 3
}


def encode_target(series):

    # If labels are already numeric, keep them.
    if pd.api.types.is_numeric_dtype(series):

        return series.astype(int)

    # Otherwise convert string labels.
    encoded = series.map(label_mapping)

    # Check for unexpected labels.
    if encoded.isna().any():

        unknown_labels = (
            series[encoded.isna()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown target labels found: {unknown_labels}"
        )

    return encoded.astype(int)


y_train = encode_target(
    train["target"]
)

y_validation = encode_target(
    validation["target"]
)

y_test = encode_target(
    test["target"]
)


print("\nEncoded target classes:")

print(
    "Train:",
    sorted(y_train.unique().tolist())
)

print(
    "Validation:",
    sorted(y_validation.unique().tolist())
)

print(
    "Test:",
    sorted(y_test.unique().tolist())
)

# ============================================================
# CLASS INFORMATION
# ============================================================

classes = [
    0,
    1,
    2,
    3
]

class_names = [
    "Natural",
    "Industrial",
    "Mining",
    "Other_Uncertain"
]


print("\nClass distribution:")

for class_id, class_name in zip(
    classes,
    class_names
):

    count = int(
        (y_train == class_id).sum()
    )

    print(
        f"{class_id} = {class_name}: {count}"
    )

# ============================================================
# CLASS-BALANCED SAMPLE WEIGHTS
# ============================================================

print("\nCalculating class-balanced sample weights...")

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

print(
    "Weight range:",
    sample_weights.min(),
    "to",
    sample_weights.max()
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

model = xgb.XGBClassifier(

    objective="multi:softprob",

    num_class=4,

    n_estimators=500,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=2,

    gamma=0,

    reg_alpha=0.0,

    reg_lambda=1.0,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("=" * 70)

model.fit(

    X_train,

    y_train,

    sample_weight=sample_weights,

    eval_set=[
        (X_train, y_train),
        (X_validation, y_validation)
    ],

    verbose=True
)


print("=" * 70)
print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

train_pred = model.predict(X_train)

validation_pred = model.predict(X_validation)

test_pred = model.predict(X_test)


train_prob = model.predict_proba(X_train)

validation_prob = model.predict_proba(X_validation)

test_prob = model.predict_proba(X_test)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision_weighted":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "recall_weighted":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "f1_weighted":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            )
    }


train_metrics = calculate_metrics(
    y_train,
    train_pred
)

validation_metrics = calculate_metrics(
    y_validation,
    validation_pred
)

test_metrics = calculate_metrics(
    y_test,
    test_pred
)


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 70)
print("TRAIN METRICS")
print("=" * 70)

for key, value in train_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


print("\n" + "=" * 70)
print("VALIDATION METRICS")
print("=" * 70)

for key, value in validation_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


print("\n" + "=" * 70)
print("TEST METRICS")
print("=" * 70)

for key, value in test_metrics.items():

    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("TEST CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    y_test,
    test_pred,
    labels=classes,
    target_names=class_names,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=classes
)

print("\n" + "=" * 70)
print("TEST CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "thermoguard_xgboost.json"
)

model.save_model(
    model_path
)

print(
    "\nModel saved:",
    model_path
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "train": train_metrics,

    "validation": validation_metrics,

    "test": test_metrics,

    "num_features": len(features),

    "num_train_samples": len(train),

    "num_validation_samples": len(validation),

    "num_test_samples": len(test),

    "classes": {
        "0": "Natural",
        "1": "Industrial",
        "2": "Mining",
        "3": "Other_Uncertain"
    }
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)


with open(
    metrics_path,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    "Metrics saved:",
    metrics_path
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.npy"
)

np.save(
    cm_path,
    cm
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nCalculating feature importance...")

importance = model.feature_importances_

importance_df = pd.DataFrame({

    "feature":
        features,

    "importance":
        importance

})

importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
)


importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


print(
    "Feature importance saved:",
    importance_path
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP 20 FEATURES")
print("=" * 70)

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test[
    [
        "satellite_id",
        "thermal_location_id",
        "latitude",
        "longitude",
        "acq_datetime"
    ]
].copy()


prediction_df["actual"] = y_test.values

prediction_df["predicted"] = test_pred

prediction_df["prob_natural"] = (
    test_prob[:, 0]
)

prediction_df["prob_industrial"] = (
    test_prob[:, 1]
)

prediction_df["prob_mining"] = (
    test_prob[:, 2]
)

prediction_df["prob_other_uncertain"] = (
    test_prob[:, 3]
)

prediction_df["prediction_confidence"] = (
    np.max(
        test_prob,
        axis=1
    )
)


prediction_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


print(
    "\nTest predictions saved:",
    prediction_path
)


# ============================================================
# SAVE TRAINING EVALUATION HISTORY
# ============================================================

evaluation_results = model.evals_result()

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w"
) as file:

    json.dump(
        evaluation_results,
        file,
        indent=4
    )


print(
    "Training history saved:",
    history_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("THERMOGUARD XGBOOST TRAINING COMPLETED")
print("=" * 70)

print(
    f"Test Accuracy : {test_metrics['accuracy']:.4f}"
)

print(
    f"Test Precision: {test_metrics['precision_weighted']:.4f}"
)

print(
    f"Test Recall   : {test_metrics['recall_weighted']:.4f}"
)

print(
    f"Test F1       : {test_metrics['f1_weighted']:.4f}"
)

print(
    f"Test Macro F1 : {test_metrics['f1_macro']:.4f}"
)

print("\nOutput directory:")

print(OUTPUT_DIR)

print("=" * 70)