"""
BioAnnotate: A platform for WSI and spatial transcriptomics annotation.
Main application entry point.
"""

import streamlit as st
import os
import sys

# Add the project directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
import config

# Import modules
from modules.auth import login, create_account, is_authenticated, logout
from modules.data_manager import list_projects, create_project, load_project, save_project
from modules.wsi_annotation import WSIAnnotator
from modules.spatial_annotation import SpatialAnnotator
from modules.ui_components import header, footer, sidebar_menu

# Set Streamlit page configuration
st.set_page_config(**config.PAGE_CONFIG)

# CSS
with open(os.path.join(config.CSS_DIR, "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Display header
    header()
    
    # Check for direct navigation request
    if st.session_state.get("direct_to_wsi", False):
        # Clear the flag
        st.session_state.direct_to_wsi = False
        # Directly display the WSI annotation page
        wsi_annotation_page()
        # Display footer
        footer()
        return

    # Sidebar menu
    menu_selection = sidebar_menu()
    
    # Authentication check
    if not is_authenticated():
        if menu_selection == "Login":
            login()
        elif menu_selection == "Create Account":
            create_account()
        else:
            # Default to login if not authenticated
            login()
        
        # Stop execution if not authenticated
        footer()
        return
    
    # Authenticated user workflows
    if menu_selection == "Dashboard":
        display_dashboard()
    elif menu_selection == "WSI Annotation":
        wsi_annotation_page()
    elif menu_selection == "Spatial Annotation":
        spatial_annotation_page()
    elif menu_selection == "Projects":
        projects_page()
    elif menu_selection == "Settings":
        settings_page()
    elif menu_selection == "Logout":
        logout()
        st.rerun()  # Restart the app to show login

    # Display footer
    footer()

def display_dashboard():
    st.title("Dashboard")
    
    # User welcome
    st.write(f"Welcome, {st.session_state.username}!")
    
    # Statistics and quick access
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Projects")
        projects = list_projects(st.session_state.username)
        
        if not projects:
            st.info("No projects found. Create your first project from the Projects menu.")
        else:
            for project in projects:
                st.write(f"📁 {project['name']}")
    
    with col2:
        st.subheader("Recent Activity")
        # Placeholder for recent activity
        st.write("No recent activity to display.")
    
    # Quick start guides
    st.subheader("Quick Start")
    
    tabs = st.tabs(["WSI Annotation", "Spatial Annotation"])
    
    with tabs[0]:
        st.write("Learn how to annotate WSI images:")
        st.markdown("""
        1. Create a new project from the Projects menu
        2. Upload WSI images to your project
        3. Use the annotation tools to mark regions of interest
        4. Save your annotations
        """)
        
        if st.button("Start WSI Annotation", key="start_wsi"):
            st.session_state.menu_selection = "WSI Annotation"
            st.rerun()
    
    with tabs[1]:
        st.write("Learn how to annotate spatial transcriptomics data:")
        st.markdown("""
        1. Create a new project from the Projects menu
        2. Upload spatial transcriptomics data to your project
        3. Use the annotation tools to select spots and classify them
        4. Save your annotations
        """)
        
        if st.button("Start Spatial Annotation", key="start_spatial"):
            st.session_state.menu_selection = "Spatial Annotation"
            st.rerun()

def wsi_annotation_page():
    st.title("WSI Annotation")
    
    # Load projects for selection
    projects = list_projects(st.session_state.username)
    
    if not projects:
        st.warning("No projects found. Please create a project first.")
        if st.button("Create Project"):
            st.session_state.menu_selection = "Projects"
            st.rerun()
        return
    
    # Project selection
    project_names = [p["name"] for p in projects]
    selected_project = st.selectbox("Select Project", project_names)
    
    # Load the selected project
    project_data = load_project(st.session_state.username, selected_project)
    
    # Initialize WSI annotator
    wsi_annotator = WSIAnnotator(project_data)
    
    # Run the annotator
    updated_project = wsi_annotator.run()
    
    # Save the project if annotations were made
    if updated_project:
        save_project(st.session_state.username, selected_project, updated_project)
        st.success("Annotations saved successfully!")

def spatial_annotation_page():
    st.title("Spatial Transcriptomics Annotation")
    
    # Load projects for selection
    projects = list_projects(st.session_state.username)
    
    if not projects:
        st.warning("No projects found. Please create a project first.")
        if st.button("Create Project"):
            st.session_state.menu_selection = "Projects"
            st.rerun()
        return
    
    # Project selection
    project_names = [p["name"] for p in projects]
    selected_project = st.selectbox("Select Project", project_names)
    
    # Load the selected project
    project_data = load_project(st.session_state.username, selected_project)
    
    # Initialize spatial annotator
    spatial_annotator = SpatialAnnotator(project_data)
    
    # Run the annotator
    updated_project = spatial_annotator.run()
    
    # Save the project if annotations were made
    if updated_project:
        save_project(st.session_state.username, selected_project, updated_project)
        st.success("Annotations saved successfully!")

def projects_page():
    st.title("Projects")
    
    # Tabs for project management
    tabs = st.tabs(["My Projects", "Create New Project"])
    
    with tabs[0]:
        st.subheader("My Projects")
        
        # List existing projects
        projects = list_projects(st.session_state.username)
        
        if not projects:
            st.info("No projects found. Create your first project in the 'Create New Project' tab.")
        else:
            # Display projects in a more detailed way
            for i, project in enumerate(projects):
                with st.expander(f"📁 {project['name']}", expanded=i==0):
                    st.write(f"**Description:** {project.get('description', 'No description')}")
                    st.write(f"**Created:** {project.get('created_at', 'Unknown')}")
                    
                    # Display files in the project
                    st.write("**Files:**")
                    files = project.get('files', [])
                    if not files:
                        st.write("No files in this project")
                    else:
                        for file in files:
                            st.write(f"- {file['name']} ({file['type']})")
                    
                    # Actions for this project
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Open", key=f"open_{i}"):
                            # Set the session state and redirect
                            st.session_state.selected_project = project['name']
                            st.session_state.menu_selection = "WSI Annotation" if any(f['type'] in config.WSI_FILE_TYPES + config.IMAGE_FILE_TYPES for f in files) else "Spatial Annotation"
                            st.rerun()
                    
                    with col2:
                        if st.button("Export", key=f"export_{i}"):
                            # Placeholder for export functionality
                            st.info("Export functionality not implemented in this demo")
                    
                    with col3:
                        if st.button("Delete", key=f"delete_{i}"):
                            # Placeholder for delete functionality
                            st.warning("Delete functionality not implemented in this demo")
    
    with tabs[1]:
        st.subheader("Create New Project")
        
        # Project creation form
        with st.form("create_project_form"):
            project_name = st.text_input("Project Name")
            project_description = st.text_area("Description")
            
            submitted = st.form_submit_button("Create Project")
            
            if submitted:
                if not project_name:
                    st.error("Project name is required")
                else:
                    # Create the project
                    success = create_project(st.session_state.username, project_name, project_description)
                    
                    if success:
                        st.success(f"Project '{project_name}' created successfully!")
                        # Redirect to the projects list
                        st.session_state.menu_selection = "Projects"
                        st.rerun()
                    else:
                        st.error(f"A project with the name '{project_name}' already exists")

def settings_page():
    st.title("Settings")
    
    # User settings
    st.subheader("User Settings")
    
    # Account information
    st.write(f"**Username:** {st.session_state.username}")
    
    # Change password form
    with st.expander("Change Password"):
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submitted = st.form_submit_button("Change Password")
            
            if submitted:
                # Placeholder for password change functionality
                st.info("Password change functionality not implemented in this demo")
    
    # Application settings
    st.subheader("Application Settings")
    
    # Theme selection
    theme = st.selectbox("Theme", ["Light", "Dark"])
    
    # Annotation settings
    st.subheader("Annotation Settings")
    
    # Color settings for annotations
    st.write("Annotation Colors")
    
    # Display color pickers for each annotation type
    cols = st.columns(3)
    colors = {}
    
    for i, (label, default_color) in enumerate(config.ANNOTATION_COLORS.items()):
        with cols[i % 3]:
            colors[label] = st.color_picker(f"{label.title()}", default_color)
    
    # Save settings button
    if st.button("Save Settings"):
        # Placeholder for settings save functionality
        st.success("Settings saved successfully!")

# Run the application
if __name__ == "__main__":
    main()