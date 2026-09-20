from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Raw data
FIRMS_RAW_DIR = RAW_DIR / "FIRMS"
FIRMS_SNPP_DIR = FIRMS_RAW_DIR / "viirs-snpp" / "2024"
FIRMS_JPSS1_DIR = FIRMS_RAW_DIR / "viirs-jpss1" / "2024"
OSM_RAW_DIR = RAW_DIR / "OSM"
SENTINEL2_RAW_DIR = RAW_DIR / "Sentinel2"

# Processed data
EVENTS_DIR = PROCESSED_DIR / "events"
PERSISTENCE_DIR = PROCESSED_DIR / "persistence"
SPATIAL_DIR = PROCESSED_DIR / "spatial"
SPATIAL_CHUNKS_DIR = SPATIAL_DIR / "chunks"

# Dataset directories
RESNET_DATA_DIR = DATASETS_DIR / "resnet"
XGBOOST_DATA_DIR = DATASETS_DIR / "xgboost"
VERIFICATION_DATA_DIR = DATASETS_DIR / "verification"

# Models
RESNET_MODEL_DIR = MODELS_DIR / "resnet"
XGBOOST_MODEL_DIR = MODELS_DIR / "xgboost"
FUSION_MODEL_DIR = MODELS_DIR / "fusion"

# Results
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
RISK_DIR = RESULTS_DIR / "risk"
SHAP_DIR = RESULTS_DIR / "shap"

# Important files
FIRMS_COMBINED = FIRMS_RAW_DIR / "FIRMS_India_2024.csv"
FIRMS_CLEAN = FIRMS_RAW_DIR / "FIRMS_India_2024_clean.csv"

OSM_PBF = OSM_RAW_DIR / "india-260911.osm.pbf"
OSM_INDUSTRIAL_GPKG = SPATIAL_DIR / "OSM_India_Industrial.gpkg"

FIRMS_SPATIAL = SPATIAL_DIR / "FIRMS_India_2024_spatial_enriched.csv"
FIRMS_PERSISTENCE = PERSISTENCE_DIR / "FIRMS_India_2024_persistence.csv"
FIRMS_EVENTS = EVENTS_DIR / "FIRMS_India_2024_events.csv"

SATELLITE_CANDIDATES = DATASETS_DIR / "satellite_candidates.csv"
SENTINEL2_IMAGES = SENTINEL2_RAW_DIR / "sentinel2_images"
SENTINEL2_METADATA = DATASETS_DIR / "sentinel2_metadata.csv"
