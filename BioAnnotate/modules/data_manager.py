"""
Data management module for handling projects, files, and annotations.
"""

import os
import json
import shutil
import uuid
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tifffile

# Try to import optional dependencies
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

# Import configuration
import config

def get_project_path(username, project_name):
    """Get the path for a project JSON file."""
    return os.path.join(config.ANNOTATIONS_DIR, username, f"{project_name}.json")

def get_project_data_path(username, project_name):
    """Get the data directory for a project."""
    return os.path.join(config.USER_UPLOADS_DIR, username, project_name)

def list_projects(username):
    """List all projects for a user."""
    user_projects_dir = os.path.join(config.ANNOTATIONS_DIR, username)
    
    # Create directory if it doesn't exist
    os.makedirs(user_projects_dir, exist_ok=True)
    
    projects = []
    
    # List all JSON files in the user's projects directory
    for filename in os.listdir(user_projects_dir):
        if filename.endswith(".json"):
            project_name = filename[:-5]  # Remove .json extension
            project_path = os.path.join(user_projects_dir, filename)
            
            try:
                with open(project_path, 'r') as f:
                    project_data = json.load(f)
                
                projects.append(project_data)
            except (json.JSONDecodeError, FileNotFoundError):
                # Skip invalid files
                continue
    
    # Sort by creation date (newest first)
    projects.sort(key=lambda p: p.get('created_at', ''), reverse=True)
    
    return projects

def create_project(username, project_name, description=''):
    """Create a new project for a user."""
    # Check if project already exists
    project_path = get_project_path(username, project_name)
    if os.path.exists(project_path):
        return False
    
    # Create project data directory
    project_data_dir = get_project_data_path(username, project_name)
    os.makedirs(project_data_dir, exist_ok=True)
    
    # Create project JSON
    project_data = {
        "name": project_name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "owner": username,
        "files": [],
        "annotations": []
    }
    
    # Save project data
    with open(project_path, 'w') as f:
        json.dump(project_data, f, indent=4)
    
    return True

def load_project(username, project_name):
    """Load a project for a user."""
    project_path = get_project_path(username, project_name)
    
    try:
        with open(project_path, 'r') as f:
            project_data = json.load(f)
        
        return project_data
    except (json.JSONDecodeError, FileNotFoundError):
        st.error(f"Error loading project: {project_name}")
        return None

def save_project(username, project_name, project_data):
    """Save a project for a user."""
    project_path = get_project_path(username, project_name)
    
    # Update the updated_at timestamp
    project_data["updated_at"] = datetime.now().isoformat()
    
    # Save project data
    with open(project_path, 'w') as f:
        json.dump(project_data, f, indent=4)
    
    return True

def upload_file(username, project_name, uploaded_file):
    """Upload a file to a project."""
    # Check if file is valid
    if uploaded_file is None:
        return None
    
    # Check file size
    if uploaded_file.size > config.MAX_FILE_SIZE:
        st.error(f"File size exceeds the maximum allowed size ({config.MAX_FILE_SIZE/1024/1024}MB)")
        return None
    
    # Check file extension
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in config.ALLOWED_EXTENSIONS:
        st.error(f"File type not supported. Allowed types: {', '.join(config.ALLOWED_EXTENSIONS)}")
        return None
    
    # Load project data
    project_data = load_project(username, project_name)
    if project_data is None:
        return None
    
    # Create a unique filename
    unique_filename = f"{uuid.uuid4().hex}.{file_ext}"
    file_path = os.path.join(get_project_data_path(username, project_name), unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Determine file type
    if file_ext in config.WSI_FILE_TYPES:
        file_type = "wsi"
    elif file_ext in config.IMAGE_FILE_TYPES:
        file_type = "image"
    elif file_ext in config.SPATIAL_FILE_TYPES:
        file_type = "spatial"
    else:
        file_type = "unknown"
    
    # Add file to project data
    file_info = {
        "name": uploaded_file.name,
        "path": file_path,
        "type": file_type,
        "size": uploaded_file.size,
        "uploaded_at": datetime.now().isoformat(),
        "id": unique_filename.split(".")[0]
    }
    
    project_data["files"].append(file_info)
    
    # Save project data
    save_project(username, project_name, project_data)
    
    return file_info

def get_file_details(username, project_name, file_id):
    """Get details for a file in a project."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return None
    
    for file_info in project_data["files"]:
        if file_info["id"] == file_id:
            return file_info
    
    return None

def load_image_file(file_path):
    """Load an image file."""
    # Get file extension
    file_ext = file_path.split(".")[-1].lower()
    
    if file_ext in config.WSI_FILE_TYPES and OPENSLIDE_AVAILABLE:
        # Load WSI file with OpenSlide
        slide = openslide.OpenSlide(file_path)
        
        # Get the dimensions
        dimensions = slide.dimensions
        
        # Get a thumbnail of the image
        thumbnail_size = (500, 500)
        thumbnail = slide.get_thumbnail(thumbnail_size)
        
        return {
            "type": "wsi",
            "slide": slide,
            "dimensions": dimensions,
            "thumbnail": thumbnail
        }
    elif file_ext in config.IMAGE_FILE_TYPES or file_ext in config.WSI_FILE_TYPES:
        # Load regular image or WSI when OpenSlide is not available
        try:
            img = Image.open(file_path)
            return {
                "type": "image",
                "image": img,
                "dimensions": img.size
            }
        except Exception as e:
            # Try loading with tifffile if PIL fails
            try:
                img = tifffile.imread(file_path)
                return {
                    "type": "image",
                    "image": img,
                    "dimensions": img.shape
                }
            except Exception as e2:
                st.error(f"Error loading image: {str(e2)}")
                return None
    else:
        st.error(f"Unsupported file format: {file_ext}")
        return None

def load_spatial_file(file_path):
    """Load a spatial transcriptomics file."""
    if not ANNDATA_AVAILABLE:
        st.error("AnnData package is required for spatial transcriptomics analysis")
        return None
    
    try:
        # Load h5ad file
        adata = anndata.read_h5ad(file_path)
        
        # Check if spatial data is present
        has_spatial = "spatial" in adata.obsm
        
        return {
            "type": "spatial",
            "adata": adata,
            "has_spatial": has_spatial,
            "dimensions": adata.shape
        }
    except Exception as e:
        st.error(f"Error loading spatial data: {str(e)}")
        return None

def create_annotation(username, project_name, file_id, annotation_data):
    """Create a new annotation for a file in a project."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return None
    
    # Create annotation with unique ID
    annotation = {
        "id": str(uuid.uuid4()),
        "file_id": file_id,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "data": annotation_data
    }
    
    project_data["annotations"].append(annotation)
    
    # Save project data
    save_project(username, project_name, project_data)
    
    return annotation

def update_annotation(username, project_name, annotation_id, annotation_data):
    """Update an existing annotation."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return None
    
    # Find the annotation
    for i, annotation in enumerate(project_data["annotations"]):
        if annotation["id"] == annotation_id:
            # Update the annotation
            project_data["annotations"][i]["data"] = annotation_data
            project_data["annotations"][i]["updated_at"] = datetime.now().isoformat()
            
            # Save project data
            save_project(username, project_name, project_data)
            
            return project_data["annotations"][i]
    
    return None

def delete_annotation(username, project_name, annotation_id):
    """Delete an annotation."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return False
    
    # Find the annotation
    for i, annotation in enumerate(project_data["annotations"]):
        if annotation["id"] == annotation_id:
            # Remove the annotation
            del project_data["annotations"][i]
            
            # Save project data
            save_project(username, project_name, project_data)
            
            return True
    
    return False

def get_file_annotations(username, project_name, file_id):
    """Get all annotations for a file."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return []
    
    # Filter annotations for the file
    file_annotations = [a for a in project_data["annotations"] if a["file_id"] == file_id]
    
    return file_annotations

def export_annotations(username, project_name, format="json"):
    """Export all annotations for a project."""
    project_data = load_project(username, project_name)
    if project_data is None:
        return None
    
    if format == "json":
        # Export as JSON
        return json.dumps(project_data["annotations"], indent=4)
    elif format == "csv":
        # Convert to DataFrame for CSV export
        annotations_list = []
        
        for annotation in project_data["annotations"]:
            # Flatten annotation data
            flat_annotation = {
                "id": annotation["id"],
                "file_id": annotation["file_id"],
                "created_at": annotation["created_at"],
                "updated_at": annotation["updated_at"]
            }
            
            # Add annotation data as flattened columns
            for key, value in annotation["data"].items():
                flat_annotation[f"data_{key}"] = json.dumps(value) if isinstance(value, (dict, list)) else value
            
            annotations_list.append(flat_annotation)
        
        # Create DataFrame
        df = pd.DataFrame(annotations_list)
        
        # Convert to CSV
        return df.to_csv(index=False)
    else:
        st.error(f"Unsupported export format: {format}")
        return None