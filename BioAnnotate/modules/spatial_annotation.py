"""
Spatial transcriptomics annotation module for annotating spatial transcriptomics data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import os
import json
from datetime import datetime
import uuid

# Try to import optional dependencies
try:
    import anndata
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False

# Import modules
from modules.data_manager import load_spatial_file, get_file_annotations, create_annotation, update_annotation, delete_annotation
import config

class SpatialAnnotator:
    def __init__(self, project_data):
        """Initialize the spatial annotator with project data."""
        self.project_data = project_data
        self.username = st.session_state.username
        self.project_name = project_data["name"]
        
        # Set up state variables if they don't exist
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize session state variables."""
        if "spatial_selected_file" not in st.session_state:
            st.session_state.spatial_selected_file = None
        
        if "spatial_view_mode" not in st.session_state:
            st.session_state.spatial_view_mode = "Spatial"
        
        if "spatial_selected_gene" not in st.session_state:
            st.session_state.spatial_selected_gene = None
        
        if "spatial_annotation_mode" not in st.session_state:
            st.session_state.spatial_annotation_mode = "Spot Selection"
        
        if "spatial_annotation_label" not in st.session_state:
            st.session_state.spatial_annotation_label = "tumor"
        
        if "spatial_annotation_color" not in st.session_state:
            st.session_state.spatial_annotation_color = "#FF0000"
        
        if "spatial_current_annotation" not in st.session_state:
            st.session_state.spatial_current_annotation = None
        
        if "spatial_temp_annotations" not in st.session_state:
            st.session_state.spatial_temp_annotations = []
        
        if "spatial_saved_annotations" not in st.session_state:
            st.session_state.spatial_saved_annotations = []
        
        if "spatial_selected_spots" not in st.session_state:
            st.session_state.spatial_selected_spots = []
        
        if "spatial_data_object" not in st.session_state:
            st.session_state.spatial_data_object = None
    
    def run(self):
        """Run the spatial annotator."""
        st.write("### Spatial Transcriptomics Annotation Tool")
        
        # Filter for spatial files
        spatial_files = [f for f in self.project_data.get("files", []) 
                       if f.get("type") == "spatial"]
        
        if not spatial_files:
            st.warning("No spatial transcriptomics files found in this project. Please upload files first.")
            self._show_file_upload()
            return self.project_data
        
        # File selection
        file_names = [f["name"] for f in spatial_files]
        selected_file_name = st.selectbox("Select Spatial File", file_names, 
                                         index=0 if st.session_state.spatial_selected_file is None else 
                                         file_names.index(next((f["name"] for f in spatial_files if f["id"] == st.session_state.spatial_selected_file), file_names[0])))
        
        # Find the selected file
        selected_file = next((f for f in spatial_files if f["name"] == selected_file_name), None)
        
        if selected_file:
            st.session_state.spatial_selected_file = selected_file["id"]
            
            # Load the annotations for this file
            file_annotations = get_file_annotations(self.username, self.project_name, selected_file["id"])
            st.session_state.spatial_saved_annotations = file_annotations
            
            # Display the annotation interface
            return self._annotation_interface(selected_file)
        
        return self.project_data
    
    def _show_file_upload(self):
        """Show the file upload interface."""
        st.write("### Upload Spatial File")
        
        uploaded_file = st.file_uploader("Upload a file", 
                                       type=config.SPATIAL_FILE_TYPES,
                                       help="Select a spatial transcriptomics file to upload (h5ad format)")
        
        if uploaded_file:
            from modules.data_manager import upload_file
            
            # Upload the file to the project
            file_info = upload_file(self.username, self.project_name, uploaded_file)
            
            if file_info:
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
                
                # Update project data
                self.project_data["files"].append(file_info)
                
                # Select the newly uploaded file
                st.session_state.spatial_selected_file = file_info["id"]
                
                # Rerun to show the file
                st.rerun()
    
    def _annotation_interface(self, file_info):
        """Main annotation interface for a selected file."""
        # Load the spatial data
        spatial_data = load_spatial_file(file_info["path"])
        
        if not spatial_data:
            st.error(f"Error loading spatial data: {file_info['name']}")
            return self.project_data
        
        # Store the spatial data object
        st.session_state.spatial_data_object = spatial_data
        
        # Create three columns for the interface
        tool_col, visual_col = st.columns([1, 3])
        
        # Tool column
        with tool_col:
            self._annotation_tools(file_info, spatial_data)
        
        # Visualization column
        with visual_col:
            self._display_spatial_data(file_info, spatial_data)
        
        # Annotation list (below the columns)
        self._annotation_list(file_info)
        
        return self.project_data
    
    def _annotation_tools(self, file_info, spatial_data):
        """Annotation tools panel."""
        st.write("### Annotation Tools")
        
        # View mode
        st.write("#### View Mode")
        view_mode = st.radio("Mode", 
                           ["Spatial", "UMAP", "Gene Expression"],
                           index=["Spatial", "UMAP", "Gene Expression"].index(st.session_state.spatial_view_mode))
        
        st.session_state.spatial_view_mode = view_mode
        
        # Gene selection for gene expression view
        if view_mode == "Gene Expression":
            # Get available genes
            adata = spatial_data["adata"]
            gene_names = adata.var_names.tolist()
            
            # Add search box for genes
            gene_search = st.text_input("Search genes")
            
            if gene_search:
                # Filter genes by search term
                filtered_genes = [g for g in gene_names if gene_search.lower() in g.lower()]
                if not filtered_genes:
                    st.warning(f"No genes found matching '{gene_search}'")
                    gene_options = gene_names[:100]  # Show first 100 genes as fallback
                else:
                    gene_options = filtered_genes[:100]  # Limit to first 100 matches
            else:
                gene_options = gene_names[:100]  # Show first 100 genes by default
            
            # Gene selection
            selected_gene = st.selectbox("Select Gene", gene_options,
                                       index=0 if st.session_state.spatial_selected_gene is None else
                                       gene_options.index(st.session_state.spatial_selected_gene) if st.session_state.spatial_selected_gene in gene_options else 0)
            
            st.session_state.spatial_selected_gene = selected_gene
        
        # Annotation mode
        st.write("#### Annotation Mode")
        annotation_mode = st.radio("Annotation Type", 
                                 ["Spot Selection", "Cell Type", "Gene Expression", "Spatial Feature"],
                                 index=["Spot Selection", "Cell Type", "Gene Expression", "Spatial Feature"].index(st.session_state.spatial_annotation_mode))
        
        st.session_state.spatial_annotation_mode = annotation_mode
        
        # Annotation label
        st.write("#### Annotation Label")
        label_options = ["tumor", "normal", "stroma", "immune", "necrosis", "other"]
        annotation_label = st.selectbox("Label", label_options,
                                      index=label_options.index(st.session_state.spatial_annotation_label))
        
        st.session_state.spatial_annotation_label = annotation_label
        
        # Annotation color
        st.write("#### Annotation Color")
        annotation_color = st.color_picker("Color", config.ANNOTATION_COLORS.get(annotation_label, "#FF0000"))
        
        st.session_state.spatial_annotation_color = annotation_color
        
        # Action buttons
        st.write("#### Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Annotation"):
                self._save_current_annotation(file_info)
        
        with col2:
            if st.button("Clear Selection"):
                st.session_state.spatial_temp_annotations = []
                st.session_state.spatial_current_annotation = None
                st.session_state.spatial_selected_spots = []
                st.rerun()
    
    def _display_spatial_data(self, file_info, spatial_data):
        """Display the spatial data with annotation capability."""
        st.write("### Data Visualization")
        
        # Get the AnnData object
        adata = spatial_data["adata"]
        
        # Create a figure based on the selected view mode
        if st.session_state.spatial_view_mode == "Spatial":
            self._spatial_view(adata)
        elif st.session_state.spatial_view_mode == "UMAP":
            self._umap_view(adata)
        elif st.session_state.spatial_view_mode == "Gene Expression":
            self._gene_expression_view(adata)
    
    def _spatial_view(self, adata):
        """Display the spatial view of the data."""
        # Check if spatial coordinates are available
        if "spatial" not in adata.obsm:
            st.error("No spatial coordinates found in the data.")
            return
        
        # Get spatial coordinates
        spatial_coords = adata.obsm["spatial"]
        
        # Create an interactive plot for spatial data
        fig = px.scatter(
            x=spatial_coords[:, 0],
            y=spatial_coords[:, 1],
            color=st.session_state.spatial_selected_gene if st.session_state.spatial_selected_gene else None,
            color_continuous_scale="viridis",
            title="Spatial Transcriptomics Data",
            labels={"x": "X", "y": "Y"},
            custom_data=[np.arange(len(spatial_coords))],  # Pass the indices as custom data
        )
        
        # Adjust marker size and appearance
        fig.update_traces(
            marker=dict(
                size=10,
                opacity=0.7,
                line=dict(width=1, color="DarkSlateGrey")
            )
        )
        
        # Add selected spots
        if st.session_state.spatial_selected_spots:
            selected_x = [spatial_coords[i, 0] for i in st.session_state.spatial_selected_spots]
            selected_y = [spatial_coords[i, 1] for i in st.session_state.spatial_selected_spots]
            
            fig.add_trace(
                go.Scatter(
                    x=selected_x,
                    y=selected_y,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=st.session_state.spatial_annotation_color,
                        line=dict(width=2, color="black")
                    ),
                    name="Selected Spots"
                )
            )
        
        # Add already saved annotations
        for annotation in st.session_state.spatial_saved_annotations:
            annotation_data = annotation["data"]
            if "spots" in annotation_data:
                spots = annotation_data["spots"]
                annot_x = [spatial_coords[i, 0] for i in spots]
                annot_y = [spatial_coords[i, 1] for i in spots]
                
                fig.add_trace(
                    go.Scatter(
                        x=annot_x,
                        y=annot_y,
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=annotation_data.get("color", "#FF0000"),
                            opacity=0.5,
                            symbol="circle-open",
                            line=dict(width=2)
                        ),
                        name=f"{annotation_data.get('label', 'Unknown')} ({len(spots)} spots)"
                    )
                )
        
        # Set the layout for better appearance
        fig.update_layout(
            height=600,
            width=800,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set callback for selection
        fig.update_layout(clickmode="event+select")
        
        # Display the figure
        selected_points = plotly_events(fig, click_event=True, override_height=600)
        
        # Process selected points
        if selected_points:
            for point in selected_points:
                point_index = point.get("customdata")[0]
                
                # Add or remove point from selection
                if point_index in st.session_state.spatial_selected_spots:
                    st.session_state.spatial_selected_spots.remove(point_index)
                else:
                    st.session_state.spatial_selected_spots.append(point_index)
            
            # Create annotation data
            if st.session_state.spatial_selected_spots:
                annotation_data = {
                    "type": st.session_state.spatial_annotation_mode,
                    "label": st.session_state.spatial_annotation_label,
                    "color": st.session_state.spatial_annotation_color,
                    "spots": st.session_state.spatial_selected_spots
                }
                
                # Update current annotation
                st.session_state.spatial_current_annotation = annotation_data
                
                # Highlight that there are unsaved spots
                if len(st.session_state.spatial_selected_spots) > 0:
                    st.info(f"{len(st.session_state.spatial_selected_spots)} spots selected. Click 'Save Annotation' to save them.")
    
    def _umap_view(self, adata):
        """Display UMAP visualization of the data."""
        # Check if UMAP is already computed
        if "X_umap" not in adata.obsm:
            # Compute UMAP if scanpy is available
            if SCANPY_AVAILABLE:
                with st.spinner("Computing UMAP..."):
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
                    sc.pp.pca(adata, n_comps=50)
                    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
                    sc.tl.umap(adata)
            else:
                st.error("Scanpy is required for UMAP visualization but not available")
                return
        
        # Get UMAP coordinates
        umap_coords = adata.obsm["X_umap"]
        
        # Create an interactive plot for UMAP
        fig = px.scatter(
            x=umap_coords[:, 0],
            y=umap_coords[:, 1],
            color=st.session_state.spatial_selected_gene if st.session_state.spatial_selected_gene else None,
            color_continuous_scale="viridis",
            title="UMAP Visualization",
            labels={"x": "UMAP1", "y": "UMAP2"},
            custom_data=[np.arange(len(umap_coords))]  # Pass the indices as custom data
        )
        
        # Adjust marker size and appearance
        fig.update_traces(
            marker=dict(
                size=8,
                opacity=0.7,
                line=dict(width=1, color="DarkSlateGrey")
            )
        )
        
        # Add selected spots
        if st.session_state.spatial_selected_spots:
            selected_x = [umap_coords[i, 0] for i in st.session_state.spatial_selected_spots]
            selected_y = [umap_coords[i, 1] for i in st.session_state.spatial_selected_spots]
            
            fig.add_trace(
                go.Scatter(
                    x=selected_x,
                    y=selected_y,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=st.session_state.spatial_annotation_color,
                        line=dict(width=2, color="black")
                    ),
                    name="Selected Spots"
                )
            )
        
        # Set the layout for better appearance
        fig.update_layout(
            height=600,
            width=800,
            template="plotly_white"
        )
        
        # Set callback for selection
        fig.update_layout(clickmode="event+select")
        
        # Display the figure
        selected_points = plotly_events(fig, click_event=True, override_height=600)
        
        # Process selected points
        if selected_points:
            for point in selected_points:
                point_index = point.get("customdata")[0]
                
                # Add or remove point from selection
                if point_index in st.session_state.spatial_selected_spots:
                    st.session_state.spatial_selected_spots.remove(point_index)
                else:
                    st.session_state.spatial_selected_spots.append(point_index)
            
            # Create annotation data
            if st.session_state.spatial_selected_spots:
                annotation_data = {
                    "type": st.session_state.spatial_annotation_mode,
                    "label": st.session_state.spatial_annotation_label,
                    "color": st.session_state.spatial_annotation_color,
                    "spots": st.session_state.spatial_selected_spots
                }
                
                # Update current annotation
                st.session_state.spatial_current_annotation = annotation_data
                
                # Highlight that there are unsaved spots
                if len(st.session_state.spatial_selected_spots) > 0:
                    st.info(f"{len(st.session_state.spatial_selected_spots)} spots selected. Click 'Save Annotation' to save them.")
    
    def _gene_expression_view(self, adata):
        """Display gene expression visualization."""
        # Check if gene is selected
        if not st.session_state.spatial_selected_gene:
            st.warning("Please select a gene to visualize expression.")
            return
        
        # Get the gene expression data
        gene = st.session_state.spatial_selected_gene
        
        # Check if gene is in the dataset
        if gene not in adata.var_names:
            st.error(f"Gene '{gene}' not found in the dataset.")
            return
        
        # Get gene expression values
        gene_expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
        
        # Get spatial coordinates
        if "spatial" in adata.obsm:
            spatial_coords = adata.obsm["spatial"]
            
            # Create an interactive plot for gene expression in spatial coordinates
            fig = px.scatter(
                x=spatial_coords[:, 0],
                y=spatial_coords[:, 1],
                color=gene_expr,
                color_continuous_scale="viridis",
                title=f"{gene} Expression in Spatial Coordinates",
                labels={"x": "X", "y": "Y", "color": "Expression"},
                custom_data=[np.arange(len(spatial_coords))]  # Pass the indices as custom data
            )
            
            # Adjust marker size and appearance
            fig.update_traces(
                marker=dict(
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color="DarkSlateGrey")
                )
            )
            
            # Add selected spots
            if st.session_state.spatial_selected_spots:
                selected_x = [spatial_coords[i, 0] for i in st.session_state.spatial_selected_spots]
                selected_y = [spatial_coords[i, 1] for i in st.session_state.spatial_selected_spots]
                
                fig.add_trace(
                    go.Scatter(
                        x=selected_x,
                        y=selected_y,
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=st.session_state.spatial_annotation_color,
                            line=dict(width=2, color="black")
                        ),
                        name="Selected Spots"
                    )
                )
        else:
            # Create histogram of gene expression
            fig = px.histogram(
                x=gene_expr,
                title=f"{gene} Expression Distribution",
                labels={"x": "Expression", "y": "Count"},
                nbins=50,
                opacity=0.7
            )
        
        # Set the layout for better appearance
        fig.update_layout(
            height=600,
            width=800,
            template="plotly_white"
        )
        
        # Set callback for selection if spatial view
        if "spatial" in adata.obsm:
            fig.update_layout(clickmode="event+select")
            
            # Display the figure
            selected_points = plotly_events(fig, click_event=True, override_height=600)
            
            # Process selected points
            if selected_points:
                for point in selected_points:
                    point_index = point.get("customdata")[0]
                    
                    # Add or remove point from selection
                    if point_index in st.session_state.spatial_selected_spots:
                        st.session_state.spatial_selected_spots.remove(point_index)
                    else:
                        st.session_state.spatial_selected_spots.append(point_index)
                
                # Create annotation data
                if st.session_state.spatial_selected_spots:
                    annotation_data = {
                        "type": st.session_state.spatial_annotation_mode,
                        "label": st.session_state.spatial_annotation_label,
                        "color": st.session_state.spatial_annotation_color,
                        "spots": st.session_state.spatial_selected_spots
                    }
                    
                    # Update current annotation
                    st.session_state.spatial_current_annotation = annotation_data
                    
                    # Highlight that there are unsaved spots
                    if len(st.session_state.spatial_selected_spots) > 0:
                        st.info(f"{len(st.session_state.spatial_selected_spots)} spots selected. Click 'Save Annotation' to save them.")
        else:
            # Just display the figure
            st.plotly_chart(fig)
    
    def _annotation_list(self, file_info):
        """Display the list of annotations."""
        st.write("### Annotations")
        
        # Show saved annotations
        if st.session_state.spatial_saved_annotations:
            # Create a table of annotations
            annotations_df = pd.DataFrame([
                {
                    "ID": a["id"],
                    "Type": a["data"].get("type", "Unknown"),
                    "Label": a["data"].get("label", "Unknown"),
                    "Spots": len(a["data"].get("spots", [])),
                    "Created": a["created_at"].split("T")[0]
                }
                for a in st.session_state.spatial_saved_annotations
            ])
            
            # Show annotations table
            st.dataframe(annotations_df, use_container_width=True)
            
            # Delete selected annotation
            col1, col2 = st.columns([3, 1])
            
            with col1:
                annotation_to_delete = st.selectbox(
                    "Select annotation",
                    options=[a["id"] for a in st.session_state.spatial_saved_annotations],
                    format_func=lambda x: next((f"{a['data'].get('label', 'Unknown')} ({a['data'].get('type', 'Unknown')}: {len(a['data'].get('spots', []))} spots)" 
                                              for a in st.session_state.spatial_saved_annotations if a["id"] == x), x)
                )
            
            with col2:
                if st.button("Delete Annotation"):
                    if delete_annotation(self.username, self.project_name, annotation_to_delete):
                        st.success("Annotation deleted successfully")
                        # Remove from session state
                        st.session_state.spatial_saved_annotations = [a for a in st.session_state.spatial_saved_annotations if a["id"] != annotation_to_delete]
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
                annotations_json = json.dumps([a["data"] for a in st.session_state.spatial_saved_annotations], indent=4)
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
                        "spots": json.dumps(a["data"].get("spots", []))
                    }
                    for a in st.session_state.spatial_saved_annotations
                ])
                
                st.download_button(
                    label="Download CSV",
                    data=annotations_df.to_csv(index=False),
                    file_name=f"{file_info['name']}_annotations.csv",
                    mime="text/csv"
                )
    
    def _save_current_annotation(self, file_info):
        """Save the current annotation."""
        # Check if there are selected spots
        if not st.session_state.spatial_selected_spots:
            st.warning("No spots selected for annotation.")
            return
        
        # Create annotation data
        annotation_data = {
            "type": st.session_state.spatial_annotation_mode,
            "label": st.session_state.spatial_annotation_label,
            "color": st.session_state.spatial_annotation_color,
            "spots": st.session_state.spatial_selected_spots
        }
        
        # Create annotation in the database
        result = create_annotation(
            self.username, 
            self.project_name, 
            file_info["id"], 
            annotation_data
        )
        
        if result:
            # Add to saved annotations
            st.session_state.spatial_saved_annotations.append(result)
            
            # Clear selections
            st.session_state.spatial_temp_annotations = []
            st.session_state.spatial_current_annotation = None
            st.session_state.spatial_selected_spots = []
            
            st.success("Annotation saved successfully")
            st.rerun()
        else:
            st.error("Error saving annotation")

# Function to handle Plotly events
def plotly_events(fig, click_event=True, override_height=None):
    """
    Custom implementation to handle Plotly events in Streamlit.
    This is a simplified version of the streamlit-plotly-events package.
    """
    import json
    from streamlit.components.v1 import html
    
    # Convert the figure to JSON
    fig_json = fig.to_json()
    
    # Set the height
    height = override_height or fig.layout.height or 600
    
    # Create a unique key for the component
    key = f"plotly_event_{uuid.uuid4().hex}"
    
    # Create HTML and JavaScript for the component
    component_html = f"""
    <div id="{key}" style="height: {height}px;"></div>
    <script type="text/javascript">
        // Initialize the figure
        var figure = {fig_json};
        var selected_points = [];
        
        // Render the figure
        Plotly.newPlot("{key}", figure.data, figure.layout, {{responsive: true}});
        
        // Add click event listener
        document.getElementById("{key}").on("plotly_click", function(data) {{
            var points = data.points;
            selected_points = points.map(p => ({{
                curveNumber: p.curveNumber,
                pointIndex: p.pointIndex,
                x: p.x,
                y: p.y,
                customdata: p.customdata
            }}));
            
            // Send event to Streamlit
            if (window.Streamlit) {{
                window.Streamlit.setComponentValue(selected_points);
            }}
        }});
    </script>
    """
    
    # Use a workaround to store the selected points in session state
    if f"{key}_selected_points" not in st.session_state:
        st.session_state[f"{key}_selected_points"] = []
    
    # Render the component
    html(component_html, height=height, width=fig.layout.width or 800)
    
    # Return the selected points
    return st.session_state.get(f"{key}_selected_points", [])