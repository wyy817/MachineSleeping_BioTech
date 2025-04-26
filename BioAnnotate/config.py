"""
Configuration settings for the BioAnnotate application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
USER_UPLOADS_DIR = os.path.join(DATA_DIR, "user_uploads")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")

# Assets
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_DIR = os.path.join(ASSETS_DIR, "css")
IMG_DIR = os.path.join(ASSETS_DIR, "img")

# Create directories if they don't exist
for directory in [DATA_DIR, SAMPLES_DIR, USER_UPLOADS_DIR, ANNOTATIONS_DIR, ASSETS_DIR, CSS_DIR, IMG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Authentication
USER_DB_PATH = os.path.join(DATA_DIR, "users.json")

# Default annotation colors
ANNOTATION_COLORS = {
    "normal": "#00FF00",     # Green
    "tumor": "#FF0000",      # Red
    "stroma": "#0000FF",     # Blue
    "immune": "#FFFF00",     # Yellow
    "necrosis": "#FF00FF",   # Magenta
    "other": "#00FFFF",      # Cyan
}

# WSI annotation types
WSI_ANNOTATION_TYPES = [
    "Region of Interest",
    "Cell Classification",
    "Tissue Classification",
    "Measurement",
    "Comment"
]

# Spatial transcriptomics annotation types
SPATIAL_ANNOTATION_TYPES = [
    "Spot Selection",
    "Gene Expression",
    "Cell Type",
    "Spatial Feature",
    "Comment"
]

# Image file types
WSI_FILE_TYPES = ["svs", "ndpi", "tiff", "tif"]
IMAGE_FILE_TYPES = ["png", "jpg", "jpeg"]
SPATIAL_FILE_TYPES = ["h5ad"]
ALLOWED_EXTENSIONS = set(WSI_FILE_TYPES + IMAGE_FILE_TYPES + SPATIAL_FILE_TYPES)

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024

# Default visualization settings
DEFAULT_POINT_SIZE = 5
DEFAULT_LINE_WIDTH = 2
DEFAULT_OPACITY = 0.7

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "BioAnnotate",
    "page_icon": "🔬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        "About": "# BioAnnotate\nA platform for WSI and spatial transcriptomics annotation."
    }
}