import os
import pandas as pd
import rasterio
from collections import Counter


from config.paths import (
    RESNET_DATA_DIR,
    SENTINEL2_IMAGES,
)
LABELS_FILE = RESNET_DATA_DIR / "resnet_full_1880_labels.csv"
IMAGE_DIR = SENTINEL2_IMAGES

df = pd.read_csv(METADATA)

print("=" * 60)
print("SENTINEL-2 DATASET INSPECTION")
print("=" * 60)

print("\nSTATUS:")
print(df["status"].value_counts())

success = df[df["status"] == "success"].copy()

print("\nSuccessful images:", len(success))

print("\n" + "=" * 60)
print("WEAK LABEL DISTRIBUTION")
print("=" * 60)
print(success["weak_label"].value_counts())

print("\n" + "=" * 60)
print("LABEL + CONFIDENCE")
print("=" * 60)
print(pd.crosstab(
    success["weak_label"],
    success["label_confidence"]
))

print("\n" + "=" * 60)
print("CLOUD PERCENTAGE")
print("=" * 60)
print(success["cloud_percentage"].describe())

print("\n" + "=" * 60)
print("INSPECTING TIFF FILES")
print("=" * 60)

band_counts = Counter()
dimensions = Counter()

corrupt = []
missing = []

for i, row in enumerate(success.itertuples(index=False), 1):

    path = row.image_path

    if not os.path.exists(path):
        missing.append(path)
        continue

    try:
        with rasterio.open(path) as src:

            band_counts[src.count] += 1
            dimensions[(src.width, src.height)] += 1

            data = src.read(1)

            if data.size == 0:
                corrupt.append(path)

    except Exception as e:
        corrupt.append((path, str(e)))

    if i % 200 == 0:
        print(f"Checked {i}/{len(success)} images")

print("\n" + "=" * 60)
print("IMAGE BAND COUNTS")
print("=" * 60)

for bands, count in band_counts.items():
    print(f"{bands} bands : {count} images")

print("\n" + "=" * 60)
print("IMAGE DIMENSIONS")
print("=" * 60)

for dims, count in dimensions.items():
    print(f"{dims[0]} x {dims[1]} : {count} images")

print("\n" + "=" * 60)
print("FILE PROBLEMS")
print("=" * 60)

print("Missing files :", len(missing))
print("Corrupt files :", len(corrupt))

if missing:
    print("\nExample missing file:")
    print(missing[0])

if corrupt:
    print("\nExample corrupt file:")
    print(corrupt[0])

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)