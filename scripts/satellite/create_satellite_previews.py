import os
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# PATHS
# ============================================================
from config.paths import (
    SENTINEL2_METADATA,
    DATASETS_DIR,
)
METADATA = SENTINEL2_METADATA

OUTPUT_DIR = DATASETS_DIR / "satellite_previews"

# ============================================================
# SETTINGS
# ============================================================

SAMPLES_PER_LABEL = 5

LABELS = [
    "Possible_Industrial",
    "Industrial_Persistent",
    "Likely_Industrial",
    "Likely_Mining",
    "Possible_Natural",
    "Unknown"
]

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(METADATA)

df = df[df["status"] == "success"].copy()

print("=" * 60)
print("CREATING SENTINEL-2 VISUAL PREVIEWS")
print("=" * 60)

print("\nSuccessful images:", len(df))

# ============================================================
# NORMALIZATION FUNCTION
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
# CREATE RGB + FALSE COLOR
# ============================================================

def create_previews(path, output_prefix):

    with rasterio.open(path) as src:

        # Our six bands are:
        # 1 = B2
        # 2 = B3
        # 3 = B4
        # 4 = B8
        # 5 = B11
        # 6 = B12

        b2 = src.read(1)
        b3 = src.read(2)
        b4 = src.read(3)
        b8 = src.read(4)

        # -------------------------
        # RGB: B4 B3 B2
        # -------------------------

        red = normalize_band(b4)
        green = normalize_band(b3)
        blue = normalize_band(b2)

        rgb = np.dstack([red, green, blue])

        rgb_img = Image.fromarray(rgb)

        rgb_path = output_prefix + "_RGB.jpg"
        rgb_img.save(rgb_path, quality=95)

        # -------------------------
        # FALSE COLOR: B8 B4 B3
        # -------------------------

        nir = normalize_band(b8)

        false_color = np.dstack([
            nir,
            red,
            green
        ])

        false_img = Image.fromarray(false_color)

        false_path = output_prefix + "_FalseColor.jpg"
        false_img.save(false_path)

        return rgb_path, false_path


# ============================================================
# SELECT REPRESENTATIVE SAMPLES
# ============================================================

selected = []

for label in LABELS:

    subset = df[df["weak_label"] == label]

    if len(subset) == 0:
        print(f"\nWARNING: No images for {label}")
        continue

    # Select evenly distributed examples
    n = min(SAMPLES_PER_LABEL, len(subset))

    sample = subset.sample(
        n=n,
        random_state=42
    )

    selected.append(sample)

    print(f"{label}: {n} samples selected")


selected_df = pd.concat(selected, ignore_index=True)

# ============================================================
# CREATE INDIVIDUAL PREVIEWS
# ============================================================

print("\nCreating individual previews...")

preview_records = []

for i, row in selected_df.iterrows():

    satellite_id = row["satellite_id"]
    label = row["weak_label"]

    path = row["image_path"]

    safe_label = label.replace(" ", "_")

    prefix = os.path.join(
        OUTPUT_DIR,
        f"{safe_label}_{satellite_id}"
    )

    try:

        rgb_path, false_path = create_previews(
            path,
            prefix
        )

        preview_records.append({
            "satellite_id": satellite_id,
            "weak_label": label,
            "rgb_path": rgb_path,
            "false_color_path": false_path
        })

        print(
            f"[{i+1}/{len(selected_df)}] "
            f"{label} - {satellite_id}"
        )

    except Exception as e:

        print(
            f"ERROR: {satellite_id} -> {e}"
        )


# ============================================================
# SAVE PREVIEW METADATA
# ============================================================

preview_df = pd.DataFrame(preview_records)

preview_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "preview_metadata.csv"
    ),
    index=False
)

print("\nCreated previews:", len(preview_df))

# ============================================================
# CREATE CONTACT SHEET
# ============================================================

print("\nCreating contact sheet...")

thumb_width = 256
thumb_height = 256

label_height = 45

columns = 5

rows = int(np.ceil(len(preview_records) / columns))

sheet_width = columns * thumb_width
sheet_height = rows * (thumb_height + label_height)

sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, record in enumerate(preview_records):

    row_idx = i // columns
    col_idx = i % columns

    x = col_idx * thumb_width
    y = row_idx * (thumb_height + label_height)

    img = Image.open(record["rgb_path"]).convert("RGB")

    img.thumbnail(
        (thumb_width, thumb_height)
    )

    sheet.paste(img, (x, y))

    text = (
        record["weak_label"]
        + "\n"
        + record["satellite_id"]
    )

    draw.text(
        (x + 5, y + thumb_height + 3),
        text,
        fill="black"
    )

contact_path = os.path.join(
    OUTPUT_DIR,
    "sentinel2_RGB_contact_sheet.jpg"
)

sheet.save(
    contact_path,
    quality=95
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print("\nOutput folder:")
print(OUTPUT_DIR)

print("\nRGB contact sheet:")
print(contact_path)

print("\nIndividual RGB + False Color images:")
print(OUTPUT_DIR)

print("\nOriginal TIFF files were NOT modified.")