import pandas as pd
import os

# ==============================
# 1. FILE PATHS
# ==============================

from config.paths import (
    FIRMS_SNPP_DIR,
    FIRMS_JPSS1_DIR,
    FIRMS_COMBINED,
)

SNPP_DIR = FIRMS_SNPP_DIR
JPSS1_DIR = FIRMS_JPSS1_DIR
OUTPUT_FILE = FIRMS_COMBINED

# ==============================
# 2. READ DATASETS
# ==============================

print("Reading SNPP dataset...")
snpp = pd.read_csv(SNPP_DIR)

print("Reading JPSS1 dataset...")
jpss1 = pd.read_csv(JPSS1_DIR)


# ==============================
# 3. DISPLAY BASIC INFORMATION
# ==============================

print("\nSNPP shape:", snpp.shape)
print("JPSS1 shape:", jpss1.shape)

print("\nSNPP columns:")
print(snpp.columns.tolist())

print("\nJPSS1 columns:")
print(jpss1.columns.tolist())


# ==============================
# 4. ADD SOURCE COLUMN
# ==============================

snpp["source_satellite"] = "SNPP"
jpss1["source_satellite"] = "JPSS1"


# ==============================
# 5. COMBINE DATASETS
# ==============================

combined = pd.concat(
    [snpp, jpss1],
    ignore_index=True
)


# ==============================
# 6. REMOVE EXACT DUPLICATES
# ==============================

before = len(combined)

combined = combined.drop_duplicates()

after = len(combined)

print("\nExact duplicate rows removed:", before - after)


# ==============================
# 7. SAVE COMBINED DATASET
# ==============================



combined.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==============================
# 8. FINAL INFORMATION
# ==============================

print("\n===================================")
print("COMBINATION COMPLETED")
print("===================================")

print("Final shape:", combined.shape)

print("\nSatellite distribution:")
print(combined["source_satellite"].value_counts())

print("\nSaved to:")
print(OUTPUT_FILE)