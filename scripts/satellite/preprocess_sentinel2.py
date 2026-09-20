import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SENTINEL2_IMAGES,
)
IMAGE_DIR = SENTINEL2_IMAGES

MANIFEST_DIR = RESNET_DATA_DIR / "resnet_final_manifests"

OUTPUT_DIR = RESNET_DATA_DIR / "preprocessed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

TARGET_SIZE = 128

# Sentinel-2 bands in each TIFF:
#
# Band 1 = B2  Blue
# Band 2 = B3  Green
# Band 3 = B4  Red
# Band 4 = B8  NIR
# Band 5 = B11 SWIR
# Band 6 = B12 SWIR

REFLECTANCE_SCALE = 10000.0


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(image_path):

    with rasterio.open(image_path) as src:

        # Check band count
        if src.count != 6:
            raise ValueError(
                f"Expected 6 bands, found {src.count}"
            )

        # Resize directly while reading
        data = src.read(
            out_shape=(
                6,
                TARGET_SIZE,
                TARGET_SIZE
            ),
            resampling=rasterio.enums.Resampling.bilinear
        )

    # Convert to float32
    data = data.astype(np.float32)

    # Sentinel-2 reflectance scaling
    data = data / REFLECTANCE_SCALE

    # Keep values in 0-1 range
    data = np.clip(
        data,
        0.0,
        1.0
    )

    # Replace invalid values
    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    return data


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    manifest_file
):

    print("\n" + "=" * 70)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 70)

    df = pd.read_csv(
        manifest_file
    )

    print(
        f"Images to process: {len(df)}"
    )

    # Output folder
    split_output = (
        OUTPUT_DIR / split_name
    )

    split_output.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_rows = []
    failed_rows = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=split_name
    ):

        satellite_id = str(
            row["satellite_id"]
        ).strip()

        image_path = Path(
            row["image_path"]
        )

        output_file = (
            split_output /
            f"{satellite_id}.npy"
        )

        try:

            # ------------------------------------------------
            # If already processed, verify it
            # ------------------------------------------------

            if output_file.exists():

                arr = np.load(
                    output_file,
                    mmap_mode="r"
                )

                if arr.shape != (
                    6,
                    TARGET_SIZE,
                    TARGET_SIZE
                ):
                    raise ValueError(
                        f"Existing tensor has "
                        f"wrong shape: {arr.shape}"
                    )

            else:

                # Process TIFF
                arr = process_image(
                    image_path
                )

                # Verify shape
                if arr.shape != (
                    6,
                    TARGET_SIZE,
                    TARGET_SIZE
                ):
                    raise ValueError(
                        f"Wrong output shape: "
                        f"{arr.shape}"
                    )

                # Save
                np.save(
                    output_file,
                    arr.astype(np.float32)
                )

            # ------------------------------------------------
            # Save metadata
            # ------------------------------------------------

            processed_rows.append({

                "satellite_id":
                    satellite_id,

                "thermal_location_id":
                    row["thermal_location_id"],

                "resnet_class":
                    row["resnet_class"],

                "resnet_label":
                    row["resnet_label"],

                "high_confidence_candidate":
                    row[
                        "high_confidence_candidate"
                    ],

                "input_path":
                    str(image_path),

                "tensor_path":
                    str(output_file),

                "shape":
                    "6x128x128"

            })

        except Exception as e:

            failed_rows.append({

                "satellite_id":
                    satellite_id,

                "image_path":
                    str(image_path),

                "error":
                    str(e)

            })

    # ========================================================
    # SAVE MANIFESTS
    # ========================================================

    processed_df = pd.DataFrame(
        processed_rows
    )

    failed_df = pd.DataFrame(
        failed_rows
    )

    processed_file = (
        OUTPUT_DIR /
        f"{split_name}_processed.csv"
    )

    failed_file = (
        OUTPUT_DIR /
        f"{split_name}_failed.csv"
    )

    processed_df.to_csv(
        processed_file,
        index=False
    )

    failed_df.to_csv(
        failed_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\nSuccessfully processed: "
        f"{len(processed_df)}"
    )

    print(
        f"Failed: "
        f"{len(failed_df)}"
    )

    return processed_df


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SENTINEL-2 128x128x6 PREPROCESSING")
print("=" * 70)

# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

train_df = process_split(
    "train",
    MANIFEST_DIR / "train_final.csv"
)

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------

validation_df = process_split(
    "validation",
    MANIFEST_DIR / "validation_final.csv"
)

# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

test_df = process_split(
    "test",
    MANIFEST_DIR / "test_final.csv"
)

# ============================================================
# COMBINE MANIFESTS
# ============================================================

all_df = pd.concat(
    [
        train_df,
        validation_df,
        test_df
    ],
    ignore_index=True
)

all_file = (
    OUTPUT_DIR /
    "all_processed.csv"
)

all_df.to_csv(
    all_file,
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREPROCESSING SUMMARY")
print("=" * 70)

print(
    f"\nTrain tensors: "
    f"{len(train_df)}"
)

print(
    f"Validation tensors: "
    f"{len(validation_df)}"
)

print(
    f"Test tensors: "
    f"{len(test_df)}"
)

print(
    f"Total tensors: "
    f"{len(all_df)}"
)

print(
    "\nExpected:"
)

print(
    "Train = 1442"
)

print(
    "Validation = 179"
)

print(
    "Test = 173"
)

print(
    "Total = 1794"
)

# ============================================================
# SAMPLE VERIFICATION
# ============================================================

if len(all_df) > 0:

    sample_path = Path(
        all_df.iloc[0]["tensor_path"]
    )

    sample = np.load(
        sample_path
    )

    print("\n" + "=" * 70)
    print("SAMPLE TENSOR VERIFICATION")
    print("=" * 70)

    print(
        f"Shape: {sample.shape}"
    )

    print(
        f"Dtype: {sample.dtype}"
    )

    print(
        f"Minimum: {sample.min():.6f}"
    )

    print(
        f"Maximum: {sample.max():.6f}"
    )

    print(
        f"Mean: {sample.mean():.6f}"
    )

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"\nOutput directory:"
)

print(
    OUTPUT_DIR
)