import os
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    RESNET_DATA_DIR,
    SENTINEL2_IMAGES,
)
LABELS_FILE = RESNET_DATA_DIR / "resnet_verified_labels.csv"
IMAGE_DIR = SENTINEL2_IMAGES
OUTPUT_DIR = RESNET_DATA_DIR / "verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("RESNET VISUAL VERIFICATION")
print("=" * 70)

print("\nTotal candidates:", len(df))

print("\nClass distribution:")
print(df["resnet_class"].value_counts())

# ============================================================
# NORMALIZE BAND
# ============================================================

def normalize_band(band):

    band = band.astype(np.float32)

    valid = band[np.isfinite(band)]

    if len(valid) == 0:
        return np.zeros_like(band, dtype=np.uint8)

    low = np.percentile(valid, 2)
    high = np.percentile(valid, 98)

    if high <= low:
        return np.zeros_like(band, dtype=np.uint8)

    band = np.clip(band, low, high)

    band = (band - low) / (high - low)

    return (band * 255).astype(np.uint8)


# ============================================================
# CREATE RGB AND FALSE COLOR
# ============================================================

def create_images(path):

    with rasterio.open(path) as src:

        # 6 bands:
        # 1 = B2 Blue
        # 2 = B3 Green
        # 3 = B4 Red
        # 4 = B8 NIR
        # 5 = B11 SWIR
        # 6 = B12 SWIR

        b2 = src.read(1)
        b3 = src.read(2)
        b4 = src.read(3)
        b8 = src.read(4)

        blue = normalize_band(b2)
        green = normalize_band(b3)
        red = normalize_band(b4)
        nir = normalize_band(b8)

        # RGB
        rgb = np.dstack([
            red,
            green,
            blue
        ])

        # False Color: NIR-Red-Green
        false_color = np.dstack([
            nir,
            red,
            green
        ])

        return (
            Image.fromarray(rgb),
            Image.fromarray(false_color)
        )


# ============================================================
# CREATE INDIVIDUAL PREVIEWS
# ============================================================

print("\nCreating individual previews...")

records = []

for i, row in df.iterrows():

    satellite_id = row["satellite_id"]
    label = row["resnet_class"]
    weak_label = row["weak_label"]

    path = row["image_path"]

    try:

        rgb_img, false_img = create_images(path)

        rgb_path = os.path.join(
            OUTPUT_DIR,
            f"{satellite_id}_RGB.jpg"
        )

        false_path = os.path.join(
            OUTPUT_DIR,
            f"{satellite_id}_FalseColor.jpg"
        )

        rgb_img.save(rgb_path, quality=95)
        false_img.save(false_path, quality=95)

        records.append({
            "satellite_id": satellite_id,
            "resnet_class": label,
            "weak_label": weak_label,
            "rgb_path": rgb_path,
            "false_color_path": false_path
        })

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(df)}")

    except Exception as e:

        print(
            f"ERROR: {satellite_id} -> {e}"
        )


preview_df = pd.DataFrame(records)

preview_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "verification_metadata.csv"
    ),
    index=False
)

print("\nSuccessfully created:", len(preview_df))


# ============================================================
# CONTACT SHEET FUNCTION
# ============================================================

def create_contact_sheet(data, image_column, output_name):

    # 8 images per row
    columns = 8

    image_size = 160
    text_height = 45

    rows = int(
        np.ceil(len(data) / columns)
    )

    sheet_width = columns * image_size

    sheet_height = rows * (
        image_size + text_height
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, (_, row) in enumerate(data.iterrows()):

        image_path = row[image_column]

        img = Image.open(
            image_path
        ).convert("RGB")

        img.thumbnail(
            (image_size, image_size)
        )

        row_number = i // columns
        column_number = i % columns

        x = column_number * image_size
        y = row_number * (
            image_size + text_height
        )

        sheet.paste(
            img,
            (x, y)
        )

        # Show ID + original weak label
        text = (
            str(row["satellite_id"])
            + "\n"
            + str(row["weak_label"])
        )

        draw.text(
            (x + 3, y + image_size + 3),
            text,
            fill="black"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    sheet.save(
        output_path,
        quality=95
    )

    print("\nCreated:")
    print(output_path)


# ============================================================
# SPLIT CLASSES
# ============================================================

industrial = preview_df[
    preview_df["resnet_class"] == "Industrial"
].copy()

non_industrial = preview_df[
    preview_df["resnet_class"] == "Non_Industrial"
].copy()

# Sort by weak label so related samples stay together
industrial = industrial.sort_values(
    "weak_label"
)

non_industrial = non_industrial.sort_values(
    "weak_label"
)

# ============================================================
# CREATE CONTACT SHEETS
# ============================================================

print("\nCreating contact sheets...")

create_contact_sheet(
    industrial,
    "rgb_path",
    "INDUSTRIAL_RGB.jpg"
)

create_contact_sheet(
    industrial,
    "false_color_path",
    "INDUSTRIAL_FALSE_COLOR.jpg"
)

create_contact_sheet(
    non_industrial,
    "rgb_path",
    "NON_INDUSTRIAL_RGB.jpg"
)

create_contact_sheet(
    non_industrial,
    "false_color_path",
    "NON_INDUSTRIAL_FALSE_COLOR.jpg"
)

# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION PREVIEWS COMPLETE")
print("=" * 70)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nFiles created:")
print("1. INDUSTRIAL_RGB.jpg")
print("2. INDUSTRIAL_FALSE_COLOR.jpg")
print("3. NON_INDUSTRIAL_RGB.jpg")
print("4. NON_INDUSTRIAL_FALSE_COLOR.jpg")

print("\nOriginal TIFF files were NOT modified.")

print("\n" + "=" * 70)