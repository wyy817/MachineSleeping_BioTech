"""
WSI annotation module for annotating whole slide images.
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io
import os
import json
from datetime import datetime
import uuid
from streamlit_drawable_canvas import st_canvas

# Try to import optional dependencies
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False

# Import modules
from modules.data_manager import load_image_file, get_file_annotations, create_annotation, update_annotation, delete_annotation
import config

class WSIAnnotator:
    def __init__(self, project_data):
        """Initialize the WSI annotator with project data."""
        self.project_data = project_data
        self.username = st.session_state.username
        self.project_name = project_data["name"]
        
        # Set up state variables if they don't exist
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "wsi_selected_file" not in st.session_state:
            st.session_state.wsi_selected_file = None
        
        if "wsi_zoom_level" not in st.session_state:
            st.session_state.wsi_zoom_level = 1.0
        
        if "wsi_position" not in st.session_state:
            st.session_state.wsi_position = (0, 0)
        
        if "wsi_annotation_mode" not in st.session_state:
            st.session_state.wsi_annotation_mode = "Region"
        
        if "wsi_annotation_color" not in st.session_state:
            st.session_state.wsi_annotation_color = "#FF0000"
        
        if "wsi_annotation_label" not in st.session_state:
            st.session_state.wsi_annotation_label = "tumor"
        
        if "wsi_current_annotation" not in st.session_state:
            st.session_state.wsi_current_annotation = None
        
        if "wsi_temp_annotations" not in st.session_state:
            st.session_state.wsi_temp_annotations = []
        
        if "wsi_saved_annotations" not in st.session_state:
            st.session_state.wsi_saved_annotations = []
    
    def run(self):
        """Run the WSI annotator."""
        st.write("### WSI Annotation Tool")
        
        # Filter for WSI and regular image files
        image_files = [f for f in self.project_data.get("files", []) 
                     if f.get("type") in ["wsi", "image"]]
        
        if not image_files:
            st.warning("No WSI or image files found in this project. Please upload files first.")
            self._show_file_upload()
            return self.project_data
        
        # File selection
        file_names = [f["name"] for f in image_files]
        selected_file_name = st.selectbox("Select Image", file_names, 
                                         index=0 if st.session_state.wsi_selected_file is None else 
                                         file_names.index(next((f["name"] for f in image_files if f["id"] == st.session_state.wsi_selected_file), file_names[0])))
        
        # Find the selected file
        selected_file = next((f for f in image_files if f["name"] == selected_file_name), None)
        
        if selected_file:
            st.session_state.wsi_selected_file = selected_file["id"]
            
            # Load the annotations for this file
            file_annotations = get_file_annotations(self.username, self.project_name, selected_file["id"])
            st.session_state.wsi_saved_annotations = file_annotations
            
            # Display the annotation interface
            return self._annotation_interface(selected_file)
        
        return self.project_data
    
    def _show_file_upload(self):
        """Show the file upload interface."""
        st.write("### Upload WSI or Image File")
        
        uploaded_file = st.file_uploader("Upload a file", 
                                       type=config.WSI_FILE_TYPES + config.IMAGE_FILE_TYPES,
                                       help="Select a WSI or regular image file to upload")
        
        if uploaded_file:
            from modules.data_manager import upload_file
            
            # Upload the file to the project
            file_info = upload_file(self.username, self.project_name, uploaded_file)
            
            if file_info:
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                
                # Update project data
                self.project_data["files"].append(file_info)
                
                # Select the newly uploaded file
                st.session_state.wsi_selected_file = file_info["id"]
                
                # Rerun to show the file
                st.rerun()
    
    def _annotation_interface(self, file_info):
        """Main annotation interface for a selected file."""
        # Create three columns for the interface
        tool_col, image_col = st.columns([1, 3])
        
        # Tool column
        with tool_col:
            self._annotation_tools(file_info)
        
        # Image column
        with image_col:
            self._display_image(file_info)
        
        # Annotation list (below the columns)
        self._annotation_list(file_info)
        
        return self.project_data
    
    def _annotation_tools(self, file_info):
        """Annotation tools panel."""
        st.write("### Annotation Tools")
        
        # Annotation mode
        st.write("#### Annotation Mode")
        annotation_mode = st.radio("Mode", 
                                 ["Region", "Point", "Polygon", "Line", "Measurement"],
                                 index=["Region", "Point", "Polygon", "Line", "Measurement"].index(st.session_state.wsi_annotation_mode))
        
        st.session_state.wsi_annotation_mode = annotation_mode
        
        # Annotation label
        st.write("#### Annotation Label")
        label_options = ["tumor", "normal", "stroma", "immune", "necrosis", "other"]
        annotation_label = st.selectbox("Label", label_options,
                                      index=label_options.index(st.session_state.wsi_annotation_label))
        
        st.session_state.wsi_annotation_label = annotation_label
        
        # Annotation color
        st.write("#### Annotation Color")
        annotation_color = st.color_picker("Color", config.ANNOTATION_COLORS.get(annotation_label, "#FF0000"))
        
        st.session_state.wsi_annotation_color = annotation_color
        
        # Image navigation
        st.write("#### Image Navigation")
        
        # Zoom level
        zoom_level = st.slider("Zoom", 0.1, 5.0, st.session_state.wsi_zoom_level, 0.1)
        if zoom_level != st.session_state.wsi_zoom_level:
            st.session_state.wsi_zoom_level = zoom_level
            st.rerun()
        
        # Position controls
        st.write("Position")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⬅️", help="Move left"):
                x, y = st.session_state.wsi_position
                st.session_state.wsi_position = (max(0, x - 100), y)
                st.rerun()
            if st.button("⬆️", help="Move up"):
                x, y = st.session_state.wsi_position
                st.session_state.wsi_position = (x, max(0, y - 100))
                st.rerun()
        
        with col2:
            if st.button("➡️", help="Move right"):
                x, y = st.session_state.wsi_position
                st.session_state.wsi_position = (x + 100, y)
                st.rerun()
            if st.button("⬇️", help="Move down"):
                x, y = st.session_state.wsi_position
                st.session_state.wsi_position = (x, y + 100)
                st.rerun()
        
        # Reset view
        if st.button("Reset View"):
            st.session_state.wsi_position = (0, 0)
            st.session_state.wsi_zoom_level = 1.0
            st.rerun()
        
        # Action buttons
        st.write("#### Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Annotation"):
                self._save_current_annotation(file_info)
        
        with col2:
            if st.button("Clear Canvas"):
                st.session_state.wsi_temp_annotations = []
                st.session_state.wsi_current_annotation = None
                st.rerun()
    
    def _display_image(self, file_info):
        """Display the image with annotation canvas (simplified version)."""
        st.write("### Image View")

        # Load the image
        image_data = load_image_file(file_info["path"])

        if not image_data:
            st.error(f"Error loading image: {file_info['name']}")
            return

        # Choose which image to show
        if image_data["type"] == "wsi" and OPENSLIDE_AVAILABLE:
            image = image_data["thumbnail"]
        else:
            image = image_data["image"]

        # Show drawing canvas
        st_canvas(
            fill_color="#FF0000",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=image,
            update_streamlit=True,
            drawing_mode="freedraw",
            key="wsi_canvas",
            display_toolbar=True,
        )

    
    def _annotation_list(self, file_info):
        """Display the list of annotations."""
        st.write("### Annotations")
        
        # Show saved annotations
        if st.session_state.wsi_saved_annotations:
            # Create a table of annotations
            annotations_df = pd.DataFrame([
                {
                    "ID": a["id"],
                    "Type": a["data"].get("type", "Unknown"),
                    "Label": a["data"].get("label", "Unknown"),
                    "Created": a["created_at"].split("T")[0]
                }
                for a in st.session_state.wsi_saved_annotations
            ])
            
            # Show annotations table
            st.dataframe(annotations_df, use_container_width=True)
            
            # Delete selected annotation
            col1, col2 = st.columns([3, 1])
            
            with col1:
                annotation_to_delete = st.selectbox(
                    "Select annotation",
                    options=[a["id"] for a in st.session_state.wsi_saved_annotations],
                    format_func=lambda x: next((f"{a['data'].get('label', 'Unknown')} ({a['data'].get('type', 'Unknown')})" 
                                              for a in st.session_state.wsi_saved_annotations if a["id"] == x), x)
                )
            
            with col2:
                if st.button("Delete Annotation"):
                    if delete_annotation(self.username, self.project_name, annotation_to_delete):
                        st.success("Annotation deleted successfully")
                        # Remove from session state
                        st.session_state.wsi_saved_annotations = [a for a in st.session_state.wsi_saved_annotations if a["id"] != annotation_to_delete]
                        st.rerun()
                    else:
                        st.error("Error deleting annotation")
        else:
            st.info("No annotations saved yet.")
        
        # Show export options
        st.write("### Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export as JSON"):
                annotations_json = json.dumps([a["data"] for a in st.session_state.wsi_saved_annotations], indent=4)
                st.download_button(
                    label="Download JSON",
                    data=annotations_json,
                    file_name=f"{file_info['name']}_annotations.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export as CSV"):
                # Create dataframe from annotations
                annotations_df = pd.DataFrame([
                    {
                        "id": a["id"],
                        "type": a["data"].get("type", "Unknown"),
                        "label": a["data"].get("label", "Unknown"),
                        "color": a["data"].get("color", "#000000"),
                        "created_at": a["created_at"],
                        "coordinates": json.dumps(a["data"].get("object", {}))
                    }
                    for a in st.session_state.wsi_saved_annotations
                ])
                
                st.download_button(
                    label="Download CSV",
                    data=annotations_df.to_csv(index=False),
                    file_name=f"{file_info['name']}_annotations.csv",
                    mime="text/csv"
                )
    
    def _save_current_annotation(self, file_info):
        """Save the current annotation."""
        # Check if there are temporary annotations to save
        if not st.session_state.wsi_temp_annotations:
            st.warning("No annotations to save.")
            return
        
        # Save each annotation
        for annotation_data in st.session_state.wsi_temp_annotations:
            # Create annotation in the database
            result = create_annotation(
                self.username, 
                self.project_name, 
                file_info["id"], 
                annotation_data
            )
            
            if result:
                # Add to saved annotations
                st.session_state.wsi_saved_annotations.append(result)
            else:
                st.error("Error saving annotation")
                return
        
        # Clear temporary annotations
        st.session_state.wsi_temp_annotations = []
        st.session_state.wsi_current_annotation = None
        
        st.success("Annotations saved successfully")
        st.rerun()
    
    def _get_drawing_mode(self):
        """Convert annotation mode to drawing mode."""
        mode_mapping = {
            "Region": "rect",
            "Point": "point",
            "Polygon": "polygon",
            "Line": "line",
            "Measurement": "line"
        }
        return mode_mapping.get(st.session_state.wsi_annotation_mode, "freedraw")
    
    def _is_new_annotation(self, object_data):
        """Check if an object is a new annotation."""
        # Check if there is a current annotation
        if not st.session_state.wsi_current_annotation:
            return True
        
        # Check if the object is already in the temporary annotations
        for annotation in st.session_state.wsi_temp_annotations:
            if annotation.get("object") == object_data:
                return False
        
        return True