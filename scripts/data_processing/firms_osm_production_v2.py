import os
import math
import pandas as pd
import geopandas as gpd

# ============================================================
# PATHS
# ============================================================

from config.paths import (
    FIRMS_CLEAN,
    OSM_INDUSTRIAL_GPKG,
    SPATIAL_CHUNKS_DIR,
)

FIRMS_FILE = FIRMS_CLEAN
OSM_FILE = OSM_INDUSTRIAL_GPKG
OUTPUT_DIR = SPATIAL_CHUNKS_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100000

# UTM zones covering the India FIRMS dataset
UTM_ZONES = range(42, 48)


# ============================================================
# OSM CATEGORIES
# ============================================================

CATEGORIES = [
    "INDUSTRIAL_AREA",
    "INDUSTRIAL",
    "INDUSTRIAL_WORKS",
    "POWER_PLANT",
    "SUBSTATION",
    "REFINERY",
    "OIL_GAS",
    "STORAGE_TANK",
    "PETROLEUM_WELL",
    "QUARRY",
    "MINING"
]


# ============================================================
# FUNCTION: GET UTM ZONE
# ============================================================

def get_utm_zone(longitude):

    zone = int(math.floor((longitude + 180) / 6) + 1)

    return max(42, min(47, zone))


# ============================================================
# FUNCTION: CLASSIFY OSM FEATURE
# ============================================================

def classify_osm(row):

    power = str(row.get("power", "")).lower()
    industrial = str(row.get("industrial", "")).lower()
    man_made = str(row.get("man_made", "")).lower()
    landuse = str(row.get("landuse", "")).lower()

    # Specific categories FIRST

    if landuse == "quarry":
        return "QUARRY"

    if man_made == "petroleum_well":
        return "PETROLEUM_WELL"

    if man_made == "storage_tank":
        return "STORAGE_TANK"

    if man_made == "mineshaft":
        return "MINING"

    if power == "plant":
        return "POWER_PLANT"

    if power == "substation":
        return "SUBSTATION"

    if industrial == "refinery":
        return "REFINERY"

    if industrial in [
        "oil",
        "gas",
        "oil_gas",
        "petroleum"
    ]:
        return "OIL_GAS"

    if man_made == "works":
        return "INDUSTRIAL_WORKS"

    if landuse == "industrial":
        return "INDUSTRIAL_AREA"

    if industrial not in ["", "nan", "none"]:
        return "INDUSTRIAL"

    return None


# ============================================================
# LOAD OSM
# ============================================================

print("\n==========================================")
print("LOADING OSM DATA")
print("==========================================")

print("Loading industrial points...")

points = gpd.read_file(
    OSM_FILE,
    layer="industrial_points"
)

print("Points loaded:", len(points))

print("Loading industrial polygons...")

polygons = gpd.read_file(
    OSM_FILE,
    layer="industrial_polygons"
)

print("Polygons loaded:", len(polygons))


# ============================================================
# COMBINE OSM
# ============================================================

print("\nCombining OSM layers...")

osm = pd.concat(
    [points, polygons],
    ignore_index=True
)

osm = gpd.GeoDataFrame(
    osm,
    geometry="geometry",
    crs="EPSG:4326"
)

print("Total OSM features:", len(osm))


# ============================================================
# CLASSIFY
# ============================================================

print("\nClassifying OSM features...")

osm["osm_category"] = osm.apply(
    classify_osm,
    axis=1
)

osm = osm[
    osm["osm_category"].isin(CATEGORIES)
].copy()

print("\nOSM category counts:")

print(
    osm["osm_category"].value_counts()
)


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

keep_columns = [
    "geometry",
    "osm_category"
]

# Keep useful identifying columns if they exist

for column in [
    "osm_id",
    "name",
    "man_made",
    "landuse",
    "industrial",
    "power"
]:

    if column in osm.columns:
        keep_columns.append(column)

osm = osm[keep_columns].copy()


# ============================================================
# PREPARE OSM FOR EACH UTM ZONE
# ============================================================

print("\n==========================================")
print("PROJECTING OSM INTO UTM ZONES")
print("==========================================")

osm_by_zone = {}

for zone in UTM_ZONES:

    epsg = 32600 + zone

    print(f"Preparing UTM zone {zone} (EPSG:{epsg})...")

    osm_zone = osm.to_crs(
        epsg=epsg
    )

    osm_by_zone[zone] = osm_zone


print("\nOSM preparation complete.")


# ============================================================
# PROCESS FIRMS IN CHUNKS
# ============================================================

print("\n==========================================")
print("STARTING FIRMS PROCESSING")
print("==========================================")

chunk_number = 0

total_rows = 0

for firms_chunk in pd.read_csv(
    FIRMS_FILE,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1

    output_file = os.path.join(
        OUTPUT_DIR,
        f"spatial_chunk_{chunk_number:04d}.csv"
    )

    # --------------------------------------------------------
    # Restart protection
    # --------------------------------------------------------

    if os.path.exists(output_file):

        print(
            f"\nChunk {chunk_number} already exists."
        )

        print(
            "Skipping:",
            output_file
        )

        total_rows += len(firms_chunk)

        continue


    print("\n------------------------------------------")
    print(f"PROCESSING CHUNK {chunk_number}")
    print("------------------------------------------")

    print(
        "Rows:",
        len(firms_chunk)
    )


    # --------------------------------------------------------
    # Create GeoDataFrame
    # --------------------------------------------------------

    firms_gdf = gpd.GeoDataFrame(
        firms_chunk.copy(),
        geometry=gpd.points_from_xy(
            firms_chunk["longitude"],
            firms_chunk["latitude"]
        ),
        crs="EPSG:4326"
    )


    # --------------------------------------------------------
    # Determine UTM zone
    # --------------------------------------------------------

    firms_gdf["utm_zone"] = firms_gdf[
        "longitude"
    ].apply(
        get_utm_zone
    )


    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    result_parts = []


    # ========================================================
    # PROCESS EACH UTM ZONE
    # ========================================================

    for zone in sorted(
        firms_gdf["utm_zone"].unique()
    ):

        print(
            f"\nProcessing UTM zone {zone}..."
        )

        firms_zone = firms_gdf[
            firms_gdf["utm_zone"] == zone
        ].copy()


        if len(firms_zone) == 0:
            continue


        epsg = 32600 + zone


        # ----------------------------------------------------
        # Project FIRMS
        # ----------------------------------------------------

        firms_zone = firms_zone.to_crs(
            epsg=epsg
        )


        # ----------------------------------------------------
        # OSM for this zone
        # ----------------------------------------------------

        osm_zone = osm_by_zone[zone]


        # ----------------------------------------------------
        # Process each category
        # ----------------------------------------------------

        for category in CATEGORIES:

            print(
                f"  Matching {category}..."
            )


            category_osm = osm_zone[
                osm_zone["osm_category"]
                == category
            ].copy()


            if len(category_osm) == 0:

                firms_zone[
                    f"distance_to_{category.lower()}_km"
                ] = float("nan")

                continue


            # ------------------------------------------------
            # Nearest spatial feature
            # ------------------------------------------------

            joined = gpd.sjoin_nearest(
                firms_zone[
                    ["geometry"]
                ],
                category_osm[
                    ["geometry"]
                ],
                how="left",
                distance_col="distance_m"
            )


            # ------------------------------------------------
            # Align distances with FIRMS rows
            # ------------------------------------------------

            distance_series = (
                joined
                .groupby(joined.index)["distance_m"]
                .min()
            )


            distance_series = (
                distance_series
                .reindex(firms_zone.index)
            )


            # Convert metres → kilometres

            firms_zone[
                f"distance_to_{category.lower()}_km"
            ] = (
                distance_series / 1000.0
            )


        # ----------------------------------------------------
        # Convert geometry back to WGS84
        # ----------------------------------------------------

        firms_zone = firms_zone.to_crs(
            epsg=4326
        )


        result_parts.append(
            firms_zone
        )


    # ========================================================
    # COMBINE ZONES
    # ========================================================

    result = pd.concat(
        result_parts,
        ignore_index=True
    )


    # Remove geometry before CSV

    result = pd.DataFrame(
        result.drop(
            columns="geometry"
        )
    )


    # ========================================================
    # PROXIMITY FLAGS
    # ========================================================

    print("\nCreating proximity features...")


    for category in CATEGORIES:

        column = (
            f"distance_to_{category.lower()}_km"
        )


        if column not in result.columns:
            continue


        # 1 km

        result[
            f"{category.lower()}_within_1km"
        ] = (
            result[column] <= 1
        ).astype("int8")


        # 5 km

        result[
            f"{category.lower()}_within_5km"
        ] = (
            result[column] <= 5
        ).astype("int8")


        # 10 km

        result[
            f"{category.lower()}_within_10km"
        ] = (
            result[column] <= 10
        ).astype("int8")


        # 25 km

        result[
            f"{category.lower()}_within_25km"
        ] = (
            result[column] <= 25
        ).astype("int8")


    # ========================================================
    # SAVE CHUNK
    # ========================================================

    result.to_csv(
        output_file,
        index=False
    )


    total_rows += len(result)


    print(
        f"\nSaved: {output_file}"
    )

    print(
        f"Rows saved: {len(result)}"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n==========================================")
print("PROCESSING COMPLETE")
print("==========================================")

print(
    "Total rows processed:",
    total_rows
)

print(
    "Output folder:",
    OUTPUT_DIR
)

print("\nNext step will be merging the chunks.")