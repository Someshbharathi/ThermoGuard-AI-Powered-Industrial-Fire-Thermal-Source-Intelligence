import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    RESNET_MODEL_DIR,
)
MANIFEST_DIR = RESNET_DATA_DIR / "resnet_final_manifests"

PREPROCESSED_DIR = RESNET_DATA_DIR / "preprocessed"

OUTPUT_DIR = RESNET_MODEL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Training parameters
# ------------------------------------------------------------

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

NUM_CLASSES = 4
NUM_CHANNELS = 6

RANDOM_SEED = 42

# Number of workers
NUM_WORKERS = 0

# Early stopping
PATIENCE = 7

# ============================================================
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = [
    "Natural",
    "Industrial",
    "Mining",
    "Other_Uncertain"
]

CLASS_TO_INDEX = {
    "Natural": 0,
    "Industrial": 1,
    "Mining": 2,
    "Other_Uncertain": 3
}

# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)

# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")

print("=" * 70)
print("THERMOGUARD - 6 CHANNEL RESNET-18")
print("=" * 70)

print(
    f"\nDevice: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA version: "
        f"{torch.version.cuda}"
    )

# ============================================================
# LOAD MANIFESTS
# ============================================================

train_manifest = (
    DATA_DIR /
    "train_processed.csv"
)

validation_manifest = (
    DATA_DIR /
    "validation_processed.csv"
)

test_manifest = (
    DATA_DIR /
    "test_processed.csv"
)

train_df = pd.read_csv(
    train_manifest
)

validation_df = pd.read_csv(
    validation_manifest
)

test_df = pd.read_csv(
    test_manifest
)

print("\nDataset sizes:")

print(
    f"Train:       {len(train_df)}"
)

print(
    f"Validation:  {len(validation_df)}"
)

print(
    f"Test:        {len(test_df)}"
)

# ============================================================
# VERIFY LABELS
# ============================================================

def convert_labels(df):

    labels = []

    for _, row in df.iterrows():

        label = row["resnet_label"]

        labels.append(
            int(label)
        )

    return np.array(
        labels,
        dtype=np.int64
    )


train_labels = convert_labels(
    train_df
)

validation_labels = convert_labels(
    validation_df
)

test_labels = convert_labels(
    test_df
)

# ============================================================
# CHANNEL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING TRAINING CHANNEL STATISTICS")
print("=" * 70)

stats_file = (
    OUTPUT_DIR /
    "channel_statistics.json"
)

if stats_file.exists():

    print(
        "\nExisting channel statistics found."
    )

    with open(
        stats_file,
        "r"
    ) as f:

        stats = json.load(f)

    channel_mean = np.array(
        stats["mean"],
        dtype=np.float32
    )

    channel_std = np.array(
        stats["std"],
        dtype=np.float32
    )

else:

    print(
        "\nCalculating from training images..."
    )

    channel_sum = np.zeros(
        NUM_CHANNELS,
        dtype=np.float64
    )

    channel_squared_sum = np.zeros(
        NUM_CHANNELS,
        dtype=np.float64
    )

    pixel_count = 0

    for tensor_path in train_df[
        "tensor_path"
    ]:

        image = np.load(
            tensor_path
        ).astype(
            np.float32
        )

        # Shape:
        # (6, 128, 128)

        pixels = image.reshape(
            NUM_CHANNELS,
            -1
        )

        channel_sum += pixels.sum(
            axis=1
        )

        channel_squared_sum += (
            (pixels ** 2).sum(axis=1)
        )

        pixel_count += (
            pixels.shape[1]
        )

    channel_mean = (
        channel_sum /
        pixel_count
    )

    variance = (
        channel_squared_sum /
        pixel_count
    ) - (
        channel_mean ** 2
    )

    channel_std = np.sqrt(
        np.maximum(
            variance,
            1e-8
        )
    )

    stats = {
        "mean": channel_mean.tolist(),
        "std": channel_std.tolist()
    }

    with open(
        stats_file,
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=4
        )

print("\nChannel statistics:")

band_names = [
    "B2_Blue",
    "B3_Green",
    "B4_Red",
    "B8_NIR",
    "B11_SWIR",
    "B12_SWIR"
]

for name, mean, std in zip(
    band_names,
    channel_mean,
    channel_std
):

    print(
        f"{name:10s} "
        f"mean={mean:.6f} "
        f"std={std:.6f}"
    )

# ============================================================
# DATASET
# ============================================================

class Sentinel2Dataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        train=False
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.train = train

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        tensor_path = Path(
            row["tensor_path"]
        )

        image = np.load(
            tensor_path
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Verify
        # ----------------------------------------------------

        if image.shape != (
            6,
            128,
            128
        ):

            raise ValueError(
                f"Invalid tensor shape: "
                f"{image.shape}"
            )

        # ----------------------------------------------------
        # Simple spatial augmentation
        # ----------------------------------------------------

        if self.train:

            # Horizontal flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=2
                ).copy()

            # Vertical flip
            if random.random() < 0.5:

                image = np.flip(
                    image,
                    axis=1
                ).copy()

            # 90-degree rotations
            k = random.randint(
                0,
                3
            )

            if k > 0:

                image = np.rot90(
                    image,
                    k=k,
                    axes=(1, 2)
                ).copy()

        # ----------------------------------------------------
        # Normalize each band
        # ----------------------------------------------------

        mean = (
            channel_mean
            .reshape(
                NUM_CHANNELS,
                1,
                1
            )
        )

        std = (
            channel_std
            .reshape(
                NUM_CHANNELS,
                1,
                1
            )
        )

        image = (
            image - mean
        ) / (
            std + 1e-8
        )

        # ----------------------------------------------------
        # Convert to Tensor
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.astype(
                np.float32
            )
        )

        label = int(
            row["resnet_label"]
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


# ============================================================
# DATASETS
# ============================================================

train_dataset = Sentinel2Dataset(
    train_df,
    train=True
)

validation_dataset = Sentinel2Dataset(
    validation_df,
    train=False
)

test_dataset = Sentinel2Dataset(
    test_df,
    train=False
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

# ============================================================
# CHECK ONE BATCH
# ============================================================

sample_images, sample_labels = next(
    iter(train_loader)
)

print("\n" + "=" * 70)
print("DATALOADER CHECK")
print("=" * 70)

print(
    f"Image batch shape: "
    f"{sample_images.shape}"
)

print(
    f"Label batch shape: "
    f"{sample_labels.shape}"
)

print(
    f"Image dtype: "
    f"{sample_images.dtype}"
)

print(
    f"Label dtype: "
    f"{sample_labels.dtype}"
)

# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for i, count in enumerate(
    class_counts
):

    print(
        f"{CLASS_NAMES[i]:18s}: "
        f"{count}"
    )

# Balanced inverse-frequency weights
class_weights = (
    len(train_labels) /
    (
        NUM_CLASSES *
        class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")

for i, weight in enumerate(
    class_weights
):

    print(
        f"{CLASS_NAMES[i]:18s}: "
        f"{weight.item():.4f}"
    )

# ============================================================
# BUILD RESNET-18
# ============================================================

print("\n" + "=" * 70)
print("BUILDING 6-CHANNEL RESNET-18")
print("=" * 70)

try:

    weights = (
        models.ResNet18_Weights.DEFAULT
    )

    model = models.resnet18(
        weights=weights
    )

    print(
        "\nImageNet pretrained weights loaded."
    )

except Exception as e:

    print(
        "\nCould not load pretrained weights."
    )

    print(
        f"Reason: {e}"
    )

    print(
        "Using randomly initialized ResNet-18."
    )

    model = models.resnet18(
        weights=None
    )

# ============================================================
# MODIFY FIRST CONVOLUTION
# ============================================================

old_conv = model.conv1

new_conv = nn.Conv2d(
    in_channels=6,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=False
)

with torch.no_grad():

    # --------------------------------------------------------
    # If pretrained:
    #
    # First 3 channels = original RGB weights
    # Last 3 channels = mean of RGB weights
    # --------------------------------------------------------

    if old_conv.weight.shape[1] == 3:

        new_conv.weight[:, :3] = (
            old_conv.weight
        )

        mean_weight = (
            old_conv.weight
            .mean(
                dim=1,
                keepdim=True
            )
        )

        new_conv.weight[:, 3:] = (
            mean_weight.repeat(
                1,
                3,
                1,
                1
            )
        )

model.conv1 = new_conv

# ============================================================
# MODIFY CLASSIFIER
# ============================================================

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(
    DEVICE
)

print(
    "\nInput channels: 6"
)

print(
    f"Output classes: {NUM_CLASSES}"
)

print(
    f"Model parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    predictions = []
    targets = []

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        preds = torch.argmax(
            outputs,
            dim=1
        )

        predictions.extend(
            preds.detach()
            .cpu()
            .numpy()
        )

        targets.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        len(train_dataset)
    )

    epoch_accuracy = accuracy_score(
        targets,
        predictions
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    predictions = []
    targets = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item()
                * images.size(0)
            )

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu()
                .numpy()
            )

            targets.extend(
                labels.cpu()
                .numpy()
            )

    loss = (
        running_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        targets,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            targets,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
        targets,
        predictions
    )


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

history = {

    "train_loss": [],
    "train_accuracy": [],

    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": []
}

best_val_loss = float(
    "inf"
)

epochs_without_improvement = 0

best_model_path = (
    OUTPUT_DIR /
    "best_resnet18_6channel.pth"
)

for epoch in range(
    NUM_EPOCHS
):

    train_loss, train_accuracy = (
        train_one_epoch()
    )

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
        _,
        _
    ) = evaluate(
        validation_loader
    )

    scheduler.step(
        val_loss
    )

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )

    history[
        "val_loss"
    ].append(
        val_loss
    )

    history[
        "val_accuracy"
    ].append(
        val_accuracy
    )

    history[
        "val_precision"
    ].append(
        val_precision
    )

    history[
        "val_recall"
    ].append(
        val_recall
    )

    history[
        "val_f1"
    ].append(
        val_f1
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        f"\nEpoch "
        f"{epoch + 1:02d}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    print(
        f"Val Precision: {val_precision:.4f} | "
        f"Val Recall: {val_recall:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate: {current_lr:.7f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "epoch":
                    epoch + 1,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,

                "class_names":
                    CLASS_NAMES,

                "channel_mean":
                    channel_mean.tolist(),

                "channel_std":
                    channel_std.tolist()
            },
            best_model_path
        )

        print(
            "✓ Best model saved."
        )

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if (
        epochs_without_improvement
        >= PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# SAVE HISTORY
# ============================================================

history_file = (
    OUTPUT_DIR /
    "training_history.json"
)

with open(
    history_file,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation loss: "
    f"{checkpoint['val_loss']:.4f}"
)

print(
    f"Best validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
    test_targets,
    test_predictions
) = evaluate(
    test_loader
)

print(
    f"\nTest Loss:       {test_loss:.4f}"
)

print(
    f"Test Accuracy:   {test_accuracy:.4f}"
)

print(
    f"Test Precision:  {test_precision:.4f}"
)

print(
    f"Test Recall:     {test_recall:.4f}"
)

print(
    f"Test F1 Score:   {test_f1:.4f}"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_targets,
    test_predictions,
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nClassification Report:")
print(report)

report_file = (
    OUTPUT_DIR /
    "classification_report.txt"
)

with open(
    report_file,
    "w"
) as f:

    f.write(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_targets,
    test_predictions
)

print("\nConfusion Matrix:")

print(cm)

np.save(
    OUTPUT_DIR /
    "confusion_matrix.npy",
    cm
)

# ============================================================
# SAVE TEST METRICS
# ============================================================

metrics = {

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "test_precision_weighted":
        float(test_precision),

    "test_recall_weighted":
        float(test_recall),

    "test_f1_weighted":
        float(test_f1),

    "test_samples":
        int(len(test_targets)),

    "best_epoch":
        int(checkpoint["epoch"])
}

with open(
    OUTPUT_DIR /
    "test_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ============================================================
# TRAINING CURVES
# ============================================================

epochs = range(
    1,
    len(history["train_loss"]) + 1
)

# ------------------------------------------------------------
# Loss
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    history["train_loss"],
    label="Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ResNet-18 Training and Validation Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "loss_curve.png",
    dpi=200
)

plt.close()

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Train Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "ResNet-18 Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "accuracy_curve.png",
    dpi=200
)

plt.close()

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = test_df.copy()

test_predictions_df[
    "predicted_class"
] = test_predictions

test_predictions_df[
    "predicted_label"
] = [
    CLASS_NAMES[p]
    for p in test_predictions
]

test_predictions_df[
    "correct"
] = (
    test_predictions_df[
        "resnet_label"
    ].values
    ==
    test_predictions
)

test_predictions_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("RESNET TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest model:"
)

print(
    best_model_path
)

print(
    "\nResults directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nSaved files:"
)

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    print(
        f"  {file.name}"
    )

print("\n" + "=" * 70)