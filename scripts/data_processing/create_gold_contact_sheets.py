import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


# ============================================================
# PATHS
# ============================================================

from config.paths import (
    VERIFICATION_DATA_DIR,
    SENTINEL2_IMAGES,
)
GOLD_TEST_FILE = VERIFICATION_DATA_DIR / "gold_test_set" / "gold_test_173.csv"

IMAGE_DIR = SENTINEL2_IMAGES

OUTPUT_DIR = VERIFICATION_DATA_DIR / "gold_test_set" / "verification_images"

CONTACT_SHEET_DIR = (
    VERIFICATION_DATA_DIR
    / "gold_test_set"
    / "contact_sheets"
)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGES_PER_SHEET = 25

THUMB_WIDTH = 220
THUMB_HEIGHT = 220

TEXT_HEIGHT = 95

CELL_WIDTH = THUMB_WIDTH
CELL_HEIGHT = THUMB_HEIGHT + TEXT_HEIGHT

COLUMNS = 5
ROWS = 5

SHEET_WIDTH = COLUMNS * CELL_WIDTH
SHEET_HEIGHT = ROWS * CELL_HEIGHT


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} samples")


# ============================================================
# FONT
# ============================================================

try:
    font = ImageFont.truetype("arial.ttf", 14)
    small_font = ImageFont.truetype("arial.ttf", 12)
except:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()


# ============================================================
# IMAGE PROCESSING
# ============================================================

def make_rgb_image(path):

    with rasterio.open(path) as src:

        # Sentinel-2:
        # B2 = Blue
        # B3 = Green
        # B4 = Red

        red = src.read(3).astype(np.float32)
        green = src.read(2).astype(np.float32)
        blue = src.read(1).astype(np.float32)

    # Stack RGB
    rgb = np.stack([red, green, blue], axis=-1)

    # Robust contrast stretch
    output = np.zeros_like(rgb)

    for i in range(3):

        band = rgb[:, :, i]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high <= low:
            output[:, :, i] = 0
        else:
            output[:, :, i] = (
                (band - low) / (high - low) * 255
            )

    output = np.clip(output, 0, 255).astype(np.uint8)

    image = Image.fromarray(output)

    image = image.resize(
        (THUMB_WIDTH, THUMB_HEIGHT),
        Image.Resampling.BILINEAR
    )

    return image


# ============================================================
# CREATE INDIVIDUAL PREVIEWS
# ============================================================

print("\nCreating image previews...")

for index, row in df.iterrows():

    satellite_id = str(row["satellite_id"])

    output_path = IMAGE_DIR / f"{satellite_id}.jpg"

    if output_path.exists():
        continue

    try:

        image = make_rgb_image(row["image_path"])

        image.save(
            output_path,
            quality=92
        )

    except Exception as e:

        print(f"ERROR {satellite_id}: {e}")


# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

print("\nCreating contact sheets...")

total = len(df)

num_sheets = math.ceil(total / IMAGES_PER_SHEET)

for sheet_number in range(num_sheets):

    start = sheet_number * IMAGES_PER_SHEET
    end = min(start + IMAGES_PER_SHEET, total)

    batch = df.iloc[start:end]

    sheet = Image.new(
        "RGB",
        (SHEET_WIDTH, SHEET_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for position, (_, row) in enumerate(batch.iterrows()):

        satellite_id = str(row["satellite_id"])

        image_path = IMAGE_DIR / f"{satellite_id}.jpg"

        if image_path.exists():

            image = Image.open(image_path)

            x = (position % COLUMNS) * CELL_WIDTH
            y = (position // COLUMNS) * CELL_HEIGHT

            sheet.paste(image, (x, y))

            # Text information
            text_x = x + 5
            text_y = y + THUMB_HEIGHT + 5

            weak_label = str(row["weak_label"])
            firms_date = str(row["firms_date"])
            image_date = str(row["image_date"])
            cloud = float(row["cloud_percentage"])

            draw.text(
                (text_x, text_y),
                satellite_id,
                fill="black",
                font=font
            )

            draw.text(
                (text_x, text_y + 18),
                f"FIRMS: {firms_date}",
                fill="black",
                font=small_font
            )

            draw.text(
                (text_x, text_y + 34),
                f"S2: {image_date}",
                fill="black",
                font=small_font
            )

            # Shorten long labels
            label_display = weak_label

            draw.text(
                (text_x, text_y + 50),
                f"Weak: {label_display}",
                fill="black",
                font=small_font
            )

            draw.text(
                (text_x, text_y + 66),
                f"Cloud: {cloud:.1f}%",
                fill="black",
                font=small_font
            )

    output_sheet = SHEET_DIR / f"batch_{sheet_number + 1:02d}.jpg"

    sheet.save(
        output_sheet,
        quality=95
    )

    print(
        f"Created batch {sheet_number + 1}/{num_sheets}: "
        f"{output_sheet}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("CONTACT SHEETS CREATED")
print("==========================================")

print(f"Samples: {total}")
print(f"Sheets: {num_sheets}")

print(f"\nImages:")
print(IMAGE_DIR)

print(f"\nContact sheets:")
print(SHEET_DIR)