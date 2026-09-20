import ee
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "sih-thermal-source-detection"

BASE_DIR = Path(__file__).resolve().parent

INPUT = BASE_DIR / "satellite_candidates.csv"
OUTPUT_DIR = BASE_DIR / "sentinel2_test"

OUTPUT_DIR.mkdir(exist_ok=True)

# Area around each FIRMS point
PATCH_SIZE = 256

# Sentinel-2 search window around FIRMS detection
DAYS_BEFORE = 15
DAYS_AFTER = 15

# Maximum cloud percentage
MAX_CLOUD = 30

# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print("=" * 65)
print("SENTINEL-2 TEST")
print("=" * 65)

print("\nConnecting to Google Earth Engine...")

ee.Initialize(project=PROJECT_ID)

print("Earth Engine connected successfully.")

# ============================================================
# LOAD CANDIDATES
# ============================================================

print("\nLoading candidate dataset...")

df = pd.read_csv(INPUT)

# Take first 5 locations
test_df = df.head(5).copy()

print(f"Testing {len(test_df)} locations.")

# ============================================================
# SENTINEL-2 COLLECTION
# ============================================================

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
)

# ============================================================
# PROCESS EACH LOCATION
# ============================================================

for index, row in test_df.iterrows():

    satellite_id = row["satellite_id"]
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    date = str(row["acq_datetime"])[:10]

    print("\n" + "-" * 65)
    print(f"Processing {satellite_id}")
    print(f"Location : {lat}, {lon}")
    print(f"FIRMS date : {date}")

    # --------------------------------------------------------
    # POINT
    # --------------------------------------------------------

    point = ee.Geometry.Point([lon, lat])

    # 1.28 km × 1.28 km approximate area
    region = point.buffer(640).bounds()

    # --------------------------------------------------------
    # DATE RANGE
    # --------------------------------------------------------

    event_date = ee.Date(date)

    start_date = event_date.advance(
        -DAYS_BEFORE,
        "day"
    )

    end_date = event_date.advance(
        DAYS_AFTER,
        "day"
    )

    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = (
        collection
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )

    count = images.size().getInfo()

    print(f"Available Sentinel-2 images: {count}")

    if count == 0:
        print("NO IMAGE FOUND - skipping.")
        continue

    # --------------------------------------------------------
    # BEST IMAGE
    # --------------------------------------------------------

    image = ee.Image(images.first())

    image_date = image.date().format(
        "YYYY-MM-dd"
    ).getInfo()

    cloud = image.get(
        "CLOUDY_PIXEL_PERCENTAGE"
    ).getInfo()

    print(f"Selected image date : {image_date}")
    print(f"Cloud percentage    : {cloud:.2f}%")

    # --------------------------------------------------------
    # SELECT BANDS
    # --------------------------------------------------------

    # B2 = Blue
    # B3 = Green
    # B4 = Red
    # B8 = NIR
    # B11 = SWIR1
    # B12 = SWIR2

    image = image.select(
        [
            "B2",
            "B3",
            "B4",
            "B8",
            "B11",
            "B12"
        ]
    )

    # --------------------------------------------------------
    # DOWNLOAD TEST IMAGE
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / (
        f"{satellite_id}.tif"
    )

    print(f"Downloading -> {output_file}")

    url = image.getDownloadURL({
        "scale": 10,
        "region": region,
        "format": "GEO_TIFF"
    })

    import requests

    response = requests.get(url)

    if response.status_code != 200:
        print(
            f"Download failed: HTTP "
            f"{response.status_code}"
        )
        continue

    with open(output_file, "wb") as f:
        f.write(response.content)

    print(
        f"Downloaded successfully "
        f"({len(response.content) / 1024:.1f} KB)"
    )

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 65)
print("5-LOCATION SENTINEL-2 TEST COMPLETE")
print("=" * 65)

print(f"\nImages saved in:")
print(OUTPUT_DIR)

print("\nNext step:")
print("Inspect the downloaded TIFF images before scaling up.")