import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    RESNET_MODEL_DIR,
    XGBOOST_DATA_DIR,
    XGBOOST_MODEL_DIR,
    PREDICTIONS_DIR,
)
XGB_TEST_FILE = XGBOOST_DATA_DIR / "test.csv"
XGB_MODEL_FILE = XGBOOST_MODEL_DIR / "thermoguard_xgboost.json"

RESNET_TEST_FILE = (
    RESNET_DATA_DIR
    / "resnet_final_manifests"
    / "test_final.csv"
)

RESNET_PREPROCESSED_DIR = RESNET_DATA_DIR / "preprocessed" / "test"

RESNET_MODEL_FILE = RESNET_MODEL_DIR / "best_resnet18_6channel.pth"

OUTPUT_DIR = PREDICTIONS_DIR / "fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CLASSES
# ============================================================

CLASS_NAMES = [
    "Natural",
    "Industrial",
    "Mining",
    "Other_Uncertain"
]

TARGET_MAP = {
    "Natural": 0,
    "Industrial": 1,
    "Mining": 2,
    "Other_Uncertain": 3
}


# ============================================================
# EXACT XGBOOST FEATURES
# ============================================================

XGB_FEATURES = [
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
# 1. XGBOOST
# ============================================================

print("=" * 60)
print("LOADING XGBOOST")
print("=" * 60)

xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(XGB_MODEL))

xgb_test = pd.read_csv(XGB_TEST)

print("XGBoost test rows:", len(xgb_test))


missing = [
    c for c in XGB_FEATURES
    if c not in xgb_test.columns
]

if missing:
    raise ValueError(
        f"Missing XGBoost features: {missing}"
    )

X_xgb = xgb_test[XGB_FEATURES].copy()

for col in XGB_FEATURES:
    X_xgb[col] = pd.to_numeric(
        X_xgb[col],
        errors="coerce"
    )

if X_xgb.isna().sum().sum() > 0:
    raise ValueError(
        "NaN values found in XGBoost input."
    )

print("\nGenerating XGBoost probabilities...")

xgb_probs = xgb_model.predict_proba(X_xgb)

print(
    "XGBoost probability shape:",
    xgb_probs.shape
)

xgb_output = pd.DataFrame({
    "satellite_id": xgb_test["satellite_id"],

    "xgb_Natural": xgb_probs[:, 0],
    "xgb_Industrial": xgb_probs[:, 1],
    "xgb_Mining": xgb_probs[:, 2],
    "xgb_Other_Uncertain": xgb_probs[:, 3]
})


# ============================================================
# 2. RESNET
# ============================================================

print("\n" + "=" * 60)
print("LOADING RESNET")
print("=" * 60)

device = torch.device("cpu")

# IMPORTANT:
# Direct torchvision ResNet.
# No custom wrapper.
resnet_model = resnet18(weights=None)

# Change 3-channel input -> 6-channel input
old_conv = resnet_model.conv1

resnet_model.conv1 = nn.Conv2d(
    6,
    old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)

# Four classes
resnet_model.fc = nn.Linear(
    resnet_model.fc.in_features,
    4
)


# ------------------------------------------------------------
# Load checkpoint
# ------------------------------------------------------------

checkpoint = torch.load(
    RESNET_MODEL,
    map_location=device
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint


print(
    "Checkpoint keys:",
    list(state_dict.keys())[:3],
    "..."
)


# ------------------------------------------------------------
# Load exact checkpoint
# ------------------------------------------------------------

resnet_model.load_state_dict(
    state_dict
)

resnet_model.to(device)
resnet_model.eval()

print("ResNet loaded successfully.")


# ============================================================
# 3. RESNET TEST DATA
# ============================================================

resnet_test = pd.read_csv(
    RESNET_TEST
)

print(
    "ResNet test rows:",
    len(resnet_test)
)


common_ids = set(
    xgb_test["satellite_id"]
).intersection(
    set(resnet_test["satellite_id"])
)

print(
    "Common test samples:",
    len(common_ids)
)


if len(common_ids) != 173:
    raise ValueError(
        "Expected 173 common test samples."
    )


# ============================================================
# 4. CHANNEL STATISTICS
# ============================================================

stats_file = (
    BASE_DIR
    / "resnet_model"
    / "channel_statistics.json"
)

with open(stats_file, "r") as f:
    stats = json.load(f)


if "mean" in stats and "std" in stats:

    mean = np.array(
        stats["mean"],
        dtype=np.float32
    )

    std = np.array(
        stats["std"],
        dtype=np.float32
    )

else:
    raise ValueError(
        "Unexpected channel_statistics.json format."
    )


print("\nChannel means:")
print(mean)

print("\nChannel std:")
print(std)


# ============================================================
# 5. FIND PREPROCESSED TENSOR
# ============================================================

def find_tensor(satellite_id):

    test_dir = (
        PREPROCESSED_DIR / "test"
    )

    path = (
        test_dir
        / f"{satellite_id}.npy"
    )

    if path.exists():
        return path

    matches = list(
        test_dir.rglob(
            f"{satellite_id}.npy"
        )
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Tensor not found for {satellite_id}"
    )


# ============================================================
# 6. RESNET PREDICTIONS
# ============================================================

print(
    "\nGenerating ResNet probabilities..."
)

resnet_rows = []

for index, row in resnet_test.iterrows():

    satellite_id = row["satellite_id"]

    tensor_path = find_tensor(
        satellite_id
    )

    array = np.load(
        tensor_path
    )

    if array.shape != (6, 128, 128):
        raise ValueError(
            f"{satellite_id} has shape "
            f"{array.shape}"
        )

    tensor = torch.from_numpy(
        array.astype(np.float32)
    )

    # Normalize exactly as during ResNet evaluation
    for channel in range(6):

        tensor[channel] = (
            tensor[channel] - mean[channel]
        ) / std[channel]

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        logits = resnet_model(
            tensor
        )

        probs = torch.softmax(
            logits,
            dim=1
        ).numpy()[0]

    resnet_rows.append({
        "satellite_id": satellite_id,

        "resnet_Natural": probs[0],
        "resnet_Industrial": probs[1],
        "resnet_Mining": probs[2],
        "resnet_Other_Uncertain": probs[3]
    })

    if (
        (index + 1) % 25 == 0
        or index + 1 == len(resnet_test)
    ):
        print(
            f"Processed "
            f"{index + 1}/"
            f"{len(resnet_test)}"
        )


resnet_output = pd.DataFrame(
    resnet_rows
)


# ============================================================
# 7. MERGE XGBOOST + RESNET
# ============================================================

print("\n" + "=" * 60)
print("ALIGNING MODELS")
print("=" * 60)

fusion = xgb_output.merge(
    resnet_output,
    on="satellite_id",
    how="inner"
)

print(
    "Fusion samples:",
    len(fusion)
)


# ============================================================
# 8. ADD TARGET
# ============================================================

target_info = xgb_test[
    [
        "satellite_id",
        "target",
        "resnet_label",
        "thermal_location_id"
    ]
]

fusion = fusion.merge(
    target_info,
    on="satellite_id",
    how="left"
)


# ============================================================
# 9. FUSION
# ============================================================

xgb_cols = [
    "xgb_Natural",
    "xgb_Industrial",
    "xgb_Mining",
    "xgb_Other_Uncertain"
]

resnet_cols = [
    "resnet_Natural",
    "resnet_Industrial",
    "resnet_Mining",
    "resnet_Other_Uncertain"
]


# 50 / 50
prob_50_50 = (
    fusion[xgb_cols].values * 0.5
    +
    fusion[resnet_cols].values * 0.5
)

fusion["fusion_50_50"] = np.argmax(
    prob_50_50,
    axis=1
)


# 70 / 30
prob_70_30 = (
    fusion[xgb_cols].values * 0.7
    +
    fusion[resnet_cols].values * 0.3
)

fusion["fusion_70_30"] = np.argmax(
    prob_70_30,
    axis=1
)


# 30 / 70
prob_30_70 = (
    fusion[xgb_cols].values * 0.3
    +
    fusion[resnet_cols].values * 0.7
)

fusion["fusion_30_70"] = np.argmax(
    prob_30_70,
    axis=1
)


# ============================================================
# 10. TARGET
# ============================================================

fusion["target_id"] = (
    fusion["target"].map(TARGET_MAP)
)

valid = fusion[
    fusion["target_id"].notna()
].copy()

y_true = (
    valid["target_id"]
    .astype(int)
    .values
)


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate(
    name,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    macro_f1 = (
        precision_recall_fscore_support(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )[2]
    )

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(
        f"Accuracy       : {accuracy:.4f}"
    )

    print(
        f"Weighted F1    : {f1:.4f}"
    )

    print(
        f"Macro F1       : {macro_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(f1),
        "macro_f1": float(macro_f1)
    }


# ============================================================
# 12. RUN
# ============================================================

results = {}

results["fusion_50_50"] = evaluate(
    "Fusion 50% XGBoost + 50% ResNet",
    valid["fusion_50_50"].values
)

results["fusion_70_30"] = evaluate(
    "Fusion 70% XGBoost + 30% ResNet",
    valid["fusion_70_30"].values
)

results["fusion_30_70"] = evaluate(
    "Fusion 30% XGBoost + 70% ResNet",
    valid["fusion_30_70"].values
)


# ============================================================
# 13. SAVE
# ============================================================

probability_file = (
    OUTPUT_DIR
    / "fusion_probabilities.csv"
)

fusion.to_csv(
    probability_file,
    index=False
)


metrics_file = (
    OUTPUT_DIR
    / "fusion_metrics.json"
)

with open(
    metrics_file,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("FUSION COMPLETE")
print("=" * 60)

print(
    "Samples evaluated:",
    len(valid)
)

print(
    "\nProbability file:"
)

print(probability_file)

print(
    "\nMetrics file:"
)

print(metrics_file)

print("\nDone.")