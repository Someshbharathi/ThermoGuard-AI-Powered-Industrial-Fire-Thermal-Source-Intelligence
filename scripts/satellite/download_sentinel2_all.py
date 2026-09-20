import ee
import pandas as pd
import requests
import time
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "sih-thermal-source-detection"

from config.paths import (
    SATELLITE_CANDIDATES,
    SENTINEL2_IMAGES,
    SENTINEL2_METADATA,
)
INPUT = SATELLITE_CANDIDATES

OUTPUT_DIR = SENTINEL2_IMAGES
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = SENTINEL2_METADATA

# Search window around FIRMS detection
DAYS_BEFORE = 15
DAYS_AFTER = 15

# Maximum scene cloud percentage
MAX_CLOUD = 30

# Image patch:
# 640 m radius -> approximately 1.28 km x 1.28 km
BUFFER_METERS = 640

# Wait between downloads
SLEEP_SECONDS = 0.5

# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("=" * 70)
print("SENTINEL-2 FULL DATASET DOWNLOAD")
print("=" * 70)

print("\nConnecting to Google Earth Engine...")

ee.Initialize(project=PROJECT_ID)

print("Earth Engine connected successfully.")

# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading satellite candidates...")

df = pd.read_csv(INPUT)

print(f"Total candidates: {len(df):,}")

# ============================================================
# LOAD PREVIOUS METADATA IF AVAILABLE
# ============================================================

if METADATA_FILE.exists():

    metadata = pd.read_csv(METADATA_FILE)

    completed_ids = set(
        metadata.loc[
            metadata["status"] == "success",
            "satellite_id"
        ].astype(str)
    )

    print(
        f"Previously completed: "
        f"{len(completed_ids):,}"
    )

else:

    metadata = pd.DataFrame(
        columns=[
            "satellite_id",
            "latitude",
            "longitude",
            "firms_date",
            "firms_datetime",
            "weak_label",
            "label_confidence",
            "image_date",
            "cloud_percentage",
            "status",
            "image_path"
        ]
    )

    completed_ids = set()

# ============================================================
# SENTINEL-2 COLLECTION
# ============================================================

collection = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            MAX_CLOUD
        )
    )
)

# ============================================================
# PROCESS EACH CANDIDATE
# ============================================================

total = len(df)

for position, (_, row) in enumerate(df.iterrows(), start=1):

    satellite_id = str(row["satellite_id"])

    # --------------------------------------------------------
    # SKIP ALREADY DOWNLOADED
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{satellite_id}.tif"

    if satellite_id in completed_ids and output_file.exists():

        print(
            f"[{position}/{total}] "
            f"{satellite_id} already completed"
        )

        continue

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    lat = float(row["latitude"])
    lon = float(row["longitude"])

    firms_date = str(row["acq_datetime"])[:10]

    weak_label = str(row["weak_label"])
    label_confidence = str(row["label_confidence"])

    print("\n" + "-" * 70)

    print(
        f"[{position}/{total}] "
        f"{satellite_id}"
    )

    print(
        f"Location: "
        f"{lat:.6f}, {lon:.6f}"
    )

    print(
        f"FIRMS date: {firms_date}"
    )

    print(
        f"Candidate label: {weak_label}"
    )

    # --------------------------------------------------------
    # CREATE REGION
    # --------------------------------------------------------

    point = ee.Geometry.Point(
        [lon, lat]
    )

    region = (
        point
        .buffer(BUFFER_METERS)
        .bounds()
    )

    # --------------------------------------------------------
    # DATE WINDOW
    # --------------------------------------------------------

    event_date = ee.Date(firms_date)

    start_date = event_date.advance(
        -DAYS_BEFORE,
        "day"
    )

    end_date = event_date.advance(
        DAYS_AFTER,
        "day"
    )

    # --------------------------------------------------------
    # FIND SENTINEL-2 IMAGES
    # --------------------------------------------------------

    images = (
        collection
        .filterBounds(region)
        .filterDate(
            start_date,
            end_date
        )
        .sort(
            "CLOUDY_PIXEL_PERCENTAGE"
        )
    )

    try:

        count = images.size().getInfo()

    except Exception as e:

        print(
            f"ERROR checking images: {e}"
        )

        continue

    print(
        f"Available images: {count}"
    )

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if count == 0:

        print("No suitable Sentinel-2 image.")

        new_row = {
            "satellite_id": satellite_id,
            "latitude": lat,
            "longitude": lon,
            "firms_date": firms_date,
            "firms_datetime": row["acq_datetime"],
            "weak_label": weak_label,
            "label_confidence": label_confidence,
            "image_date": "",
            "cloud_percentage": "",
            "status": "no_image",
            "image_path": ""
        }

        metadata = pd.concat(
            [
                metadata,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        metadata.to_csv(
            METADATA_FILE,
            index=False
        )

        continue

    # --------------------------------------------------------
    # SELECT BEST IMAGE
    # --------------------------------------------------------

    image = ee.Image(
        images.first()
    )

    image_date = (
        image.date()
        .format("YYYY-MM-dd")
        .getInfo()
    )

    cloud_percentage = image.get(
        "CLOUDY_PIXEL_PERCENTAGE"
    ).getInfo()

    print(
        f"Selected date: "
        f"{image_date}"
    )

    print(
        f"Cloud percentage: "
        f"{cloud_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # SELECT SIX BANDS
    # --------------------------------------------------------

    image = image.select(
        [
            "B2",   # Blue
            "B3",   # Green
            "B4",   # Red
            "B8",   # NIR
            "B11",  # SWIR 1
            "B12"   # SWIR 2
        ]
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    print(
        f"Downloading -> {output_file}"
    )

    try:

        url = image.getDownloadURL(
            {
                "scale": 10,
                "region": region,
                "format": "GEO_TIFF"
            }
        )

        response = requests.get(
            url,
            timeout=120
        )

        if response.status_code != 200:

            print(
                f"Download failed: "
                f"HTTP {response.status_code}"
            )

            continue

        with open(
            output_file,
            "wb"
        ) as file:

            file.write(
                response.content
            )

        size_kb = (
            output_file.stat().st_size
            / 1024
        )

        print(
            f"Downloaded successfully: "
            f"{size_kb:.1f} KB"
        )

        # ----------------------------------------------------
        # SAVE METADATA
        # ----------------------------------------------------

        new_row = {

            "satellite_id":
                satellite_id,

            "latitude":
                lat,

            "longitude":
                lon,

            "firms_date":
                firms_date,

            "firms_datetime":
                row["acq_datetime"],

            "weak_label":
                weak_label,

            "label_confidence":
                label_confidence,

            "image_date":
                image_date,

            "cloud_percentage":
                cloud_percentage,

            "status":
                "success",

            "image_path":
                str(output_file)
        }

        metadata = pd.concat(
            [
                metadata,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

        metadata.to_csv(
            METADATA_FILE,
            index=False
        )

    except Exception as e:

        print(
            f"Download error: {e}"
        )

    time.sleep(
        SLEEP_SECONDS
    )

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("SENTINEL-2 DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"\nImages folder:"
    f"\n{OUTPUT_DIR}"
)

print(
    f"\nMetadata:"
    f"\n{METADATA_FILE}"
)

if METADATA_FILE.exists():

    final_metadata = pd.read_csv(
        METADATA_FILE
    )

    print("\nStatus summary:")

    print(
        final_metadata[
            "status"
        ].value_counts()
    )

print("\nDONE.")