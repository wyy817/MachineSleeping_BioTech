"""
UI components module for reusable UI elements.
"""

import streamlit as st
import os
from PIL import Image
from datetime import datetime

# Import configuration
import config

def header():
    """Display the application header."""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Logo
        logo_path = os.path.join(config.IMG_DIR, "logo.png")
        if os.path.exists(logo_path):
            logo = Image.open(logo_path)
            st.image(logo, width=100)
        else:
            # Placeholder text if logo doesn't exist
            st.markdown("🔬")
    
    with col2:
        st.title("BioAnnotate")
        st.markdown("*WSI and Spatial Transcriptomics Annotation Platform*")
    
    st.markdown("---")

def footer():
    """Display the application footer."""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("© 2023 BioAnnotate")
    
    with col2:
        st.markdown(f"Version 1.0.0")
    
    with col3:
        st.markdown("<div style='text-align: right'>Current time: " + 
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                  unsafe_allow_html=True)

def sidebar_menu():
    """Display the sidebar menu and return the selected menu item."""
    with st.sidebar:
        st.markdown("## Navigation")
        
        # Show different menu based on authentication status
        if "authenticated" in st.session_state and st.session_state.authenticated:
            menu_options = [
                "Dashboard",
                "WSI Annotation",
                "Spatial Annotation",
                "Projects",
                "Settings",
                "Logout"
            ]
            
            # Set default selected option based on session state
            default_index = 0
            if "menu_selection" in st.session_state:
                if st.session_state.menu_selection in menu_options:
                    default_index = menu_options.index(st.session_state.menu_selection)
                # Reset the selection to avoid loops
                del st.session_state.menu_selection
            
            selected = st.radio("", menu_options, index=default_index)

            # Only reset the selection after it has been used
            if "menu_selection" in st.session_state:
                del st.session_state.menu_selection
        else:
            menu_options = ["Login", "Create Account"]
            
            # Set default selected option
            default_index = 0
            if "menu_selection" in st.session_state:
                if st.session_state.menu_selection in menu_options:
                    default_index = menu_options.index(st.session_state.menu_selection)
                # Reset the selection
                del st.session_state.menu_selection
                
            selected = st.radio("", menu_options, index=default_index)
        
        st.markdown("---")
        
        # Help and information
        with st.expander("Help & Information"):
            st.markdown("""
            **BioAnnotate** is a platform for annotating WSI and spatial transcriptomics data.
            
            - Use **WSI Annotation** to annotate whole slide images.
            - Use **Spatial Annotation** to annotate spatial transcriptomics data.
            - Manage your projects in the **Projects** section.
            """)
            
            st.markdown("For more information, check out the [documentation](https://github.com/yourusername/bioannotate).")
        
        # Display app information
        st.markdown("---")
        st.markdown("### About")
        st.markdown("BioAnnotate v1.0.0")
        st.markdown("A platform for biomedical image and data annotation.")
    
    return selected

def show_file_info(file_info):
    """Display information about a file."""
    st.write("### File Information")
    
    # Create an expander for file details
    with st.expander("File Details", expanded=False):
        st.write(f"**Name:** {file_info['name']}")
        st.write(f"**Type:** {file_info['type'].upper()}")
        st.write(f"**Size:** {file_info['size'] / 1024 / 1024:.2f} MB")
        st.write(f"**Uploaded:** {file_info['uploaded_at'].split('T')[0]}")
        
        # Show file path for debugging (can be removed in production)
        st.write(f"**Path:** {file_info['path']}")

def show_loading_spinner(text="Loading..."):
    """Display a loading spinner with custom text."""
    with st.spinner(text):
        st.empty()

def show_notification(message, type="info"):
    """Display a notification message.
    
    Args:
        message: The message to display
        type: The type of notification (info, success, warning, error)
    """
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)

def confirm_dialog(message, confirm_button="Confirm", cancel_button="Cancel"):
    """Display a confirmation dialog.
    
    Args:
        message: The message to display
        confirm_button: Text for the confirm button
        cancel_button: Text for the cancel button
        
    Returns:
        bool: True if confirmed, False otherwise
    """
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**{message}**")
    
    with col2:
        cancel = st.button(cancel_button)
    
    with col3:
        confirm = st.button(confirm_button)
    
    return confirm and not cancel

def user_info_card():
    """Display user information card."""
    if "authenticated" in st.session_state and st.session_state.authenticated:
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"### User: {st.session_state.username}")
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            st.markdown(f"**Session started:** {timestamp}")

def upload_widget(upload_types, help_text=""):
    """Display a file upload widget.
    
    Args:
        upload_types: List of allowed file extensions
        help_text: Help text to display
        
    Returns:
        uploaded_file: The uploaded file object
    """
    return st.file_uploader("Upload a file", type=upload_types, help=help_text)

def annotation_progress_bar(current, total):
    """Display an annotation progress bar.
    
    Args:
        current: Current progress
        total: Total progress
    """
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.markdown(f"**Progress:** {current}/{total} ({progress*100:.1f}%)")