import pandas as pd
import numpy as np
import rasterio
from pathlib import Path

# ============================================================
# FILES
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SENTINEL2_IMAGES,
)
LABELS_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels_with_location.csv"
IMAGE_DIR = SENTINEL2_IMAGES
OUTPUT_FILE = RESNET_DATA_DIR / "resnet_full_quality_report.csv"

# ============================================================
# QUALITY THRESHOLDS
# ============================================================

# Percentage of pixels that are invalid / no-data
EXCLUDE_BAD_PIXEL_PERCENT = 50.0

# Percentage of scene cloudiness
REVIEW_CLOUD_PERCENT = 20.0

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FULL SENTINEL-2 DATASET QUALITY CHECK")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nTotal images: {len(df)}")

# ============================================================
# QUALITY CHECK FUNCTION
# ============================================================

def inspect_image(image_path):

    result = {
        "band_count": None,
        "width": None,
        "height": None,
        "bad_pixel_percent": None,
        "zero_pixel_percent": None,
        "nan_pixel_percent": None,
        "quality_status": "ERROR",
        "quality_reason": ""
    }

    try:

        with rasterio.open(image_path) as src:

            result["band_count"] = src.count
            result["width"] = src.width
            result["height"] = src.height

            # Read all six bands
            data = src.read()

            # ------------------------------------------------
            # NaN / infinite values
            # ------------------------------------------------

            invalid = ~np.isfinite(data)

            # ------------------------------------------------
            # Zero pixels
            # ------------------------------------------------

            zero_pixels = np.all(
                data == 0,
                axis=0
            )

            # ------------------------------------------------
            # Bad pixels
            # ------------------------------------------------

            bad_pixels = invalid.any(axis=0) | zero_pixels

            total_pixels = bad_pixels.size

            bad_percent = (
                bad_pixels.sum()
                / total_pixels
                * 100
            )

            zero_percent = (
                zero_pixels.sum()
                / total_pixels
                * 100
            )

            nan_percent = (
                invalid.any(axis=0).sum()
                / total_pixels
                * 100
            )

            result["bad_pixel_percent"] = bad_percent
            result["zero_pixel_percent"] = zero_percent
            result["nan_pixel_percent"] = nan_percent

            # ------------------------------------------------
            # Band check
            # ------------------------------------------------

            if src.count != 6:

                result["quality_status"] = "EXCLUDE"

                result["quality_reason"] = (
                    f"Expected 6 bands, found {src.count}"
                )

                return result

            # ------------------------------------------------
            # Bad pixel threshold
            # ------------------------------------------------

            if bad_percent >= EXCLUDE_BAD_PIXEL_PERCENT:

                result["quality_status"] = "EXCLUDE"

                result["quality_reason"] = (
                    f"Too many bad/no-data pixels "
                    f"({bad_percent:.2f}%)"
                )

                return result

            # ------------------------------------------------
            # Cloud threshold
            # ------------------------------------------------

            # Cloud percentage is checked outside this function.
            # Here the image itself is valid.
            result["quality_status"] = "KEEP"

            result["quality_reason"] = "Good quality"

    except Exception as e:

        result["quality_status"] = "EXCLUDE"

        result["quality_reason"] = (
            f"Read error: {str(e)[:150]}"
        )

    return result


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

for i, row in df.iterrows():

    result = inspect_image(
        row["image_path"]
    )

    results.append(result)

    if (i + 1) % 100 == 0:
        print(
            f"Checked {i + 1}/{len(df)}"
        )


# ============================================================
# ADD RESULTS
# ============================================================

quality_df = pd.DataFrame(results)

df["band_count"] = quality_df["band_count"]
df["width"] = quality_df["width"]
df["height"] = quality_df["height"]

df["bad_pixel_percent"] = (
    quality_df["bad_pixel_percent"]
)

df["zero_pixel_percent"] = (
    quality_df["zero_pixel_percent"]
)

df["nan_pixel_percent"] = (
    quality_df["nan_pixel_percent"]
)

df["quality_status"] = (
    quality_df["quality_status"]
)

df["quality_reason"] = (
    quality_df["quality_reason"]
)

# ============================================================
# APPLY CLOUD REVIEW
# ============================================================

cloud_review = (
    (df["quality_status"] == "KEEP")
    &
    (df["cloud_percentage"] >= REVIEW_CLOUD_PERCENT)
)

df.loc[
    cloud_review,
    "quality_status"
] = "REVIEW"

df.loc[
    cloud_review,
    "quality_reason"
] = (
    "High scene cloud percentage "
    + df.loc[
        cloud_review,
        "cloud_percentage"
    ].round(2).astype(str)
    + "%"
)

# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("QUALITY RESULTS")
print("=" * 70)

print(
    df["quality_status"]
    .value_counts()
)

# ============================================================
# QUALITY REASONS
# ============================================================

print("\n" + "=" * 70)
print("QUALITY REASONS")
print("=" * 70)

print(
    df["quality_reason"]
    .value_counts()
)

# ============================================================
# BAND CHECK
# ============================================================

print("\n" + "=" * 70)
print("BAND COUNT")
print("=" * 70)

print(
    df["band_count"]
    .value_counts()
    .sort_index()
)

# ============================================================
# IMAGE DIMENSIONS
# ============================================================

print("\n" + "=" * 70)
print("IMAGE DIMENSIONS")
print("=" * 70)

print(
    df.groupby(
        ["width", "height"]
    ).size()
    .sort_values(
        ascending=False
    )
    .head(20)
)

# ============================================================
# BAD PIXEL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BAD PIXEL STATISTICS")
print("=" * 70)

print(
    df["bad_pixel_percent"]
    .describe()
)

# ============================================================
# CLOUD STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("CLOUD STATISTICS")
print("=" * 70)

print(
    df["cloud_percentage"]
    .describe()
)

# ============================================================
# REVIEW / EXCLUDE IMAGES
# ============================================================

problematic = df[
    df["quality_status"] != "KEEP"
][
    [
        "satellite_id",
        "resnet_class",
        "weak_label",
        "cloud_percentage",
        "bad_pixel_percent",
        "width",
        "height",
        "quality_status",
        "quality_reason"
    ]
]

print("\n" + "=" * 70)
print("IMAGES REQUIRING REVIEW / EXCLUSION")
print("=" * 70)

if len(problematic) == 0:

    print("No problematic images found.")

else:

    print(
        problematic.to_string(
            index=False
        )
    )

# ============================================================
# CLASS × QUALITY
# ============================================================

print("\n" + "=" * 70)
print("CLASS × QUALITY STATUS")
print("=" * 70)

print(
    pd.crosstab(
        df["resnet_class"],
        df["quality_status"]
    )
)

# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CHECK COMPLETE")
print("=" * 70)

print(f"\nOutput:")
print(OUTPUT_FILE)

print("\nOriginal Sentinel-2 TIFF files were NOT modified.")

print("\n" + "=" * 70)