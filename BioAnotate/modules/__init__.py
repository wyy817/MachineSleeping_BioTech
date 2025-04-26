"""
Modules package for BioAnnotate.
"""

from .auth import login, create_account, is_authenticated, logout
from .data_manager import (
    list_projects, create_project, load_project, save_project,
    upload_file, load_image_file, load_spatial_file
)
from .wsi_annotation import WSIAnnotator
from .spatial_annotation import SpatialAnnotator
from .ui_components import header, footer, sidebar_menu
from .utils import (
    generate_unique_id, get_timestamp, ensure_directory_exists,
    is_valid_file_extension, create_thumbnail, image_to_bytes,
    bytes_to_image, draw_annotations_on_image
)

__all__ = [
    # Auth
    'login', 'create_account', 'is_authenticated', 'logout',
    
    # Data Manager
    'list_projects', 'create_project', 'load_project', 'save_project',
    'upload_file', 'load_image_file', 'load_spatial_file',
    
    # Annotators
    'WSIAnnotator', 'SpatialAnnotator',
    
    # UI Components
    'header', 'footer', 'sidebar_menu',
    
    # Utils
    'generate_unique_id', 'get_timestamp', 'ensure_directory_exists',
    'is_valid_file_extension', 'create_thumbnail', 'image_to_bytes',
    'bytes_to_image', 'draw_annotations_on_image'
]
