import os
import numpy as np
import pandas as pd
import rasterio

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SENTINEL2_IMAGES,
)
LABELS_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels.csv"
IMAGE_DIR = SENTINEL2_IMAGES
QUALITY_REPORT = RESNET_DATA_DIR / "resnet_full_quality_report.csv"
OUTPUT_FILE = RESNET_DATA_DIR / "resnet_verified_labels.csv"

# ============================================================
# QUALITY THRESHOLDS
# ============================================================

# If more than this percentage of pixels are invalid/black,
# exclude the image.
MAX_BAD_PIXEL_PERCENT = 10.0

# High scene cloud does NOT automatically mean bad patch.
# Such images are marked REVIEW.
REVIEW_CLOUD_PERCENT = 20.0

EXPECTED_BANDS = 6

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("RESNET IMAGE QUALITY FILTER")
print("=" * 70)

print("\nTotal candidates:", len(df))

results = []

# ============================================================
# INSPECT EACH IMAGE
# ============================================================

for i, row in df.iterrows():

    path = row["image_path"]

    quality_status = "KEEP"
    reason = "Good quality"

    file_exists = os.path.exists(path)

    if not file_exists:

        quality_status = "EXCLUDE"
        reason = "File missing"

        results.append({
            **row.to_dict(),
            "quality_status": quality_status,
            "quality_reason": reason,
            "band_count": None,
            "width": None,
            "height": None,
            "bad_pixel_percent": None
        })

        continue

    try:

        with rasterio.open(path) as src:

            band_count = src.count
            width = src.width
            height = src.height

            # ------------------------------------------------
            # BAND CHECK
            # ------------------------------------------------

            if band_count != EXPECTED_BANDS:

                quality_status = "EXCLUDE"
                reason = f"Expected 6 bands, found {band_count}"

            # ------------------------------------------------
            # READ ALL BANDS
            # ------------------------------------------------

            data = src.read().astype(np.float32)

            # ------------------------------------------------
            # INVALID VALUES
            # ------------------------------------------------

            invalid_mask = ~np.isfinite(data)

            # ------------------------------------------------
            # BLACK / NO-DATA PIXELS
            #
            # A pixel is considered bad if ALL six bands
            # are zero or invalid.
            # ------------------------------------------------

            zero_mask = np.all(data == 0, axis=0)

            invalid_pixel_mask = np.any(
                invalid_mask,
                axis=0
            )

            bad_pixel_mask = (
                zero_mask |
                invalid_pixel_mask
            )

            total_pixels = bad_pixel_mask.size

            bad_pixels = bad_pixel_mask.sum()

            bad_pixel_percent = (
                bad_pixels / total_pixels
            ) * 100

            # ------------------------------------------------
            # QUALITY DECISION
            # ------------------------------------------------

            if bad_pixel_percent > MAX_BAD_PIXEL_PERCENT:

                quality_status = "EXCLUDE"

                reason = (
                    f"Too many bad/no-data pixels "
                    f"({bad_pixel_percent:.2f}%)"
                )

            elif (
                row["cloud_percentage"] >=
                REVIEW_CLOUD_PERCENT
            ):

                quality_status = "REVIEW"

                reason = (
                    f"High scene cloud percentage "
                    f"({row['cloud_percentage']:.2f}%)"
                )

            elif quality_status != "EXCLUDE":

                quality_status = "KEEP"

                reason = "Good quality"

    except Exception as e:

        quality_status = "EXCLUDE"

        reason = f"TIFF read error: {str(e)}"

        band_count = None
        width = None
        height = None
        bad_pixel_percent = None

    results.append({
        **row.to_dict(),
        "quality_status": quality_status,
        "quality_reason": reason,
        "band_count": band_count,
        "width": width,
        "height": height,
        "bad_pixel_percent": bad_pixel_percent
    })

    if (i + 1) % 20 == 0:
        print(
            f"Checked {i + 1}/{len(df)}"
        )

# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("QUALITY FILTER RESULTS")
print("=" * 70)

print(
    result_df["quality_status"].value_counts()
)

print("\nQuality reasons:")

print(
    result_df["quality_reason"].value_counts()
)

# ============================================================
# BAD PIXEL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("BAD / NO-DATA PIXEL STATISTICS")
print("=" * 70)

print(
    result_df["bad_pixel_percent"].describe()
)

# ============================================================
# CLOUD REVIEW
# ============================================================

review = result_df[
    result_df["quality_status"] == "REVIEW"
]

print("\n" + "=" * 70)
print("IMAGES REQUIRING REVIEW")
print("=" * 70)

print("Review count:", len(review))

if len(review) > 0:

    print(
        review[
            [
                "satellite_id",
                "weak_label",
                "cloud_percentage",
                "bad_pixel_percent",
                "quality_reason"
            ]
        ].to_string(index=False)
    )

# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# FINAL COUNTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    result_df[
        "quality_status"
    ].value_counts()
)

print("\nOutput file:")
print(OUTPUT_CSV)

print("\nOriginal TIFF files were NOT modified.")

print("\n" + "=" * 70)
print("QUALITY FILTER COMPLETE")
print("=" * 70)