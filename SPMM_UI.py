import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import time
import hashlib
import json
from PIL import Image
import io
import base64
import smtplib
import random
import string
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re



# Set page configuration
st.set_page_config(
    page_title="SPMM - Spatial Transcriptomics Multimodal Model",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #3366ff;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4d4d4d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-container {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dashboard-header {
        font-size: 1.8rem;
        color: #3366ff;
        margin-bottom: 1rem;
    }
    .dashboard-subheader {
        font-size: 1.2rem;
        color: #4d4d4d;
        margin-bottom: 1.5rem;
    }
    .feature-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .centered-text {
        text-align: center;
    }
    .logo-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
        margin-bottom: 1rem;
    }
    .result-container {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .metrics-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    .metric-box {
        text-align: center;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        width: 120px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3366ff;
    }
    .metric-name {
        font-size: 0.9rem;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# Create directories for storing data if they don't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/users"):
    os.makedirs("data/users")

# Function to create credentials file if it doesn't exist
def initialize_credentials():
    if not os.path.exists("data/credentials.json"):
        # Create with a default admin account
        default_credentials = {
            "admin": hashlib.sha256("admin123".encode()).hexdigest()
        }
        with open("data/credentials.json", "w") as f:
            json.dump(default_credentials, f)

# Initialize credentials
initialize_credentials()

# Function to generate verification code
def generate_verification_code(length=6):
    """Generate a random verification code of specified length"""
    return ''.join(random.choices(string.digits, k=length))

# Function to validate email format
def is_valid_email(email):
    """Check if the email format is valid"""
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

# Function to send verification email
def send_verification_email(email, verification_code):
    """Send verification code to user's email"""
    # These would be your actual email credentials
    # For security, you should use environment variables
    sender_email = "your_email@example.com"  # Replace with your email
    sender_password = "your_password"        # Replace with your password
    
    # For demonstration purposes only
    # In a real application, use actual email sending
    if st.session_state.get('demo_mode', False):
        st.session_state['email_sent'] = True
        st.session_state['verification_code'] = verification_code
        return True
    
    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = email
        message["Subject"] = "SPMM Account Verification"
        
        # Create message body
        body = f"""
        Hello,
        
        Thank you for registering with SPMM (Spatio-Based Pathology MultiModal).
        
        Your verification code is: {verification_code}
        
        This code will expire in 15 minutes.
        
        Best regards,
        SPMM Team
        """
        
        message.attach(MIMEText(body, "plain"))
        
        # Connect to server and send email
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, message.as_string())
        server.quit()
        
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

# Function to save user data
def save_user_data(username, email, password, is_verified=False):
    """Save user data to a JSON file"""
    user_file = f"data/users/{username}.json"
    
    # Hash the password for security
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    user_data = {
        "username": username,
        "email": email,
        "password": hashed_password,
        "is_verified": is_verified,
        "registration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": ""
    }
    
    with open(user_file, "w") as f:
        json.dump(user_data, f)

def verify_login(username, password):
    """Verify user login credentials and check verification status"""
    user_file = f"data/users/{username}.json"
    
    if not os.path.exists(user_file):
        # Check in credentials.json for legacy users
        with open("data/credentials.json", "r") as f:
            credentials = json.load(f)
        
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        if username in credentials and credentials[username] == hashed_password:
            return True
            
        return False
    
    with open(user_file, "r") as f:
        user_data = json.load(f)
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    if user_data["password"] == hashed_password:
        # Check if user is verified
        if not user_data["is_verified"]:
            st.warning("Your account is not verified. Please check your email for the verification code.")
            # Store the username for verification
            st.session_state['reg_username'] = username
            st.session_state['reg_email'] = user_data["email"]
            st.session_state['show_login'] = False
            st.session_state['show_registration'] = True
            st.session_state['registration_stage'] = 'verification'
            
            # Generate a new verification code
            verification_code = generate_verification_code()
            st.session_state['verification_code'] = verification_code
            
            # Send verification email
            send_verification_email(user_data["email"], verification_code)
            return False
        
        # Update last login time
        user_data["last_login"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(user_file, "w") as f:
            json.dump(user_data, f)
        
        return True
    
    return False

# Function to add a new user
def add_user(username, password, is_admin=False):
    with open("data/credentials.json", "r") as f:
        credentials = json.load(f)
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    credentials[username] = hashed_password
    
    with open("data/credentials.json", "w") as f:
        json.dump(credentials, f)

# Function to check if username exists
def username_exists(username):
    """Check if a username already exists"""
    if os.path.exists(f"data/users/{username}.json"):
        return True
    
    # Also check in credentials.json for legacy users
    with open("data/credentials.json", "r") as f:
        credentials = json.load(f)
    
    return username in credentials

# Function to check if email exists
def email_exists(email):
    """Check if an email is already registered"""
    for filename in os.listdir("data/users"):
        if filename.endswith(".json"):
            with open(f"data/users/{filename}", "r") as f:
                user_data = json.load(f)
                if user_data["email"] == email:
                    return True
    return False

# Function to verify user with code
def verify_user(username, verification_code):
    """Verify user with verification code"""
    if 'verification_code' in st.session_state and st.session_state['verification_code'] == verification_code:
        user_file = f"data/users/{username}.json"
        if os.path.exists(user_file):
            with open(user_file, "r") as f:
                user_data = json.load(f)
            
            user_data["is_verified"] = True
            
            with open(user_file, "w") as f:
                json.dump(user_data, f)
            
            return True
    
    return False

# Function to display registration page
def display_registration_page():
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.write("Create a new account:")
    
    # Initialize session state variables if they don't exist
    if 'registration_stage' not in st.session_state:
        st.session_state['registration_stage'] = 'form'
    if 'reg_username' not in st.session_state:
        st.session_state['reg_username'] = ""
    if 'reg_email' not in st.session_state:
        st.session_state['reg_email'] = ""
    
    # Registration form
    if st.session_state['registration_stage'] == 'form':
        username = st.text_input("Username", value=st.session_state['reg_username'])
        email = st.text_input("Email", value=st.session_state['reg_email'])
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            # Validate inputs
            if not username or not email or not password or not confirm_password:
                st.error("All fields are required")
            elif not is_valid_email(email):
                st.error("Please enter a valid email address")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif username_exists(username):
                st.error("Username already exists")
            elif email_exists(email):
                st.error("Email already registered")
            else:
                # Store username and email in session state
                st.session_state['reg_username'] = username
                st.session_state['reg_email'] = email
                
                # Generate verification code
                verification_code = generate_verification_code()
                st.session_state['verification_code'] = verification_code
                
                # Save user data (unverified)
                save_user_data(username, email, password, is_verified=False)
                
                # Send verification email
                email_sent = send_verification_email(email, verification_code)
                
                if email_sent or st.session_state.get('demo_mode', False):
                    st.session_state['registration_stage'] = 'verification'
                    st.success(f"Verification code sent to {email}")
                    st.rerun()
                else:
                    st.error("Failed to send verification email. Please try again.")
        
        # Add a link to go back to login
        if st.button("Already have an account? Log in"):
            st.session_state['registration_stage'] = 'form'
            st.session_state['reg_username'] = ""
            st.session_state['reg_email'] = ""
            st.session_state['show_login'] = True
            st.session_state['show_registration'] = False
            st.rerun()
    
    # Verification code form
    elif st.session_state['registration_stage'] == 'verification':
        st.write(f"A verification code has been sent to {st.session_state['reg_email']}")
        
        # In demo mode, show the verification code for testing
        if st.session_state.get('demo_mode', False):
            st.info(f"Demo Mode: Your verification code is {st.session_state['verification_code']}")
        
        verification_code = st.text_input("Enter verification code")
        
        if st.button("Verify"):
            if verify_user(st.session_state['reg_username'], verification_code):
                st.success("Account verified successfully!")
                st.session_state['registration_stage'] = 'form'
                st.session_state['reg_username'] = ""
                st.session_state['reg_email'] = ""
                st.session_state['show_login'] = True
                st.session_state['show_registration'] = False
                st.rerun()
            else:
                st.error("Invalid verification code")
        
        if st.button("Resend Code"):
            verification_code = generate_verification_code()
            st.session_state['verification_code'] = verification_code
            email_sent = send_verification_email(st.session_state['reg_email'], verification_code)
            
            if email_sent or st.session_state.get('demo_mode', False):
                st.success(f"Verification code resent to {st.session_state['reg_email']}")
            else:
                st.error("Failed to send verification email. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to load sample data
def load_sample_data():
    # Generate some sample spatial data for demonstration
    np.random.seed(42)
    n_spots = 200
    
    # Spatial coordinates
    coords = np.random.rand(n_spots, 2) * 100
    
    # Cell types (0-4)
    cell_types = np.random.randint(0, 5, n_spots)
    
    # Gene expression (100 genes)
    gene_expr = np.random.randn(n_spots, 100) * 0.5 + 2
    gene_expr = np.abs(gene_expr)  # Make all values positive
    
    # Create a few cluster patterns
    for i in range(n_spots):
        if coords[i, 0] < 30 and coords[i, 1] < 30:
            gene_expr[i, :20] *= 3  # Region with high expression of first 20 genes
            cell_types[i] = 0
        elif coords[i, 0] > 70 and coords[i, 1] > 70:
            gene_expr[i, 20:40] *= 3  # Region with high expression of next 20 genes
            cell_types[i] = 1
        elif coords[i, 0] > 70 and coords[i, 1] < 30:
            gene_expr[i, 40:60] *= 3  # Region with high expression of next 20 genes
            cell_types[i] = 2
        elif coords[i, 0] < 30 and coords[i, 1] > 70:
            gene_expr[i, 60:80] *= 3  # Region with high expression of next 20 genes
            cell_types[i] = 3
        else:
            gene_expr[i, 80:] *= np.random.rand() * 2 + 0.5  # Random expression of last 20 genes
            cell_types[i] = 4
    
    # Generate random gene names
    gene_names = [f"Gene_{i}" for i in range(100)]
    
    return {
        "coords": coords,
        "cell_types": cell_types,
        "gene_expr": gene_expr,
        "gene_names": gene_names
    }

# Function to generate a visualization of spatial data
def visualize_spatial_data(coords, values, title, colormap='viridis', size=50):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=values, cmap=colormap, s=size, alpha=0.8)
    plt.colorbar(scatter, label='Value')
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# Function to visualize a heatmap
def visualize_heatmap(data, row_labels, col_labels, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=False, cmap='viridis', xticklabels=col_labels, yticklabels=row_labels)
    plt.title(title)
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return buf

# Mock function to simulate model prediction
def predict_with_model(gene_expr, coords):
    # In a real scenario, this would use your actual model
    # For demo, just perform a simple classification
    n_samples = gene_expr.shape[0]
    
    # Simulated classification (0-3, from README.md: Normal, Benign, Precancerous, Malignant)
    pred_classes = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Simple rules for classification
        if gene_expr[i, :20].mean() > 5:  # If high expression of first 20 genes
            pred_classes[i] = 3  # Malignant
        elif gene_expr[i, 20:40].mean() > 5:  # If high expression of genes 20-40
            pred_classes[i] = 2  # Precancerous
        elif gene_expr[i, 40:60].mean() > 5:  # If high expression of genes 40-60
            pred_classes[i] = 1  # Benign
        else:
            pred_classes[i] = 0  # Normal
    
    # Simulated prediction probabilities
    probs = np.zeros((n_samples, 4))
    for i in range(n_samples):
        # Base probabilities
        probs[i] = np.array([0.1, 0.1, 0.1, 0.1])
        # Higher probability for predicted class
        probs[i, pred_classes[i]] = 0.7
        # Normalize to sum to 1
        probs[i] /= probs[i].sum()
    
    # Simulate some features
    features = np.random.randn(n_samples, 10)
    
    return {
        "classification": probs,
        "features": features,
        "pred_classes": pred_classes
    }

# Main application
def main():
    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ""

    # Initialize registration flow control
    if 'show_registration' not in st.session_state:
        st.session_state['show_registration'] = False
    if 'show_login' not in st.session_state:
        st.session_state['show_login'] = True

    # Login page
    if not st.session_state.logged_in:
        # Logo and header
        st.markdown('<div class="centered-text">', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">SPMM</h1>', unsafe_allow_html=True)
        st.markdown('<h2 class="sub-header">Spatio-Based Pathology MultiModal</h2>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show login or registration based on state
        if st.session_state['show_login']:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            st.write("Please login to access the SPMM model:")
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Login"):
                    if verify_login(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            
            with col2:
                if st.button("Register"):
                    st.session_state['show_login'] = False
                    st.session_state['show_registration'] = True
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif st.session_state['show_registration']:
            display_registration_page()
        
        # Information about the model
        st.markdown('---')
        st.markdown('### About SPMM')
        st.write("""
        The Spatio-Based Pathology MultiModal (SPMM) is a state-of-the-art multimodal medical foundation model 
        specifically designed for tumor/cancer slice prediction tasks. It integrates spatial transcriptomic data, 
        pathological images, and clinical text information to provide accurate tumor classification, staging, 
        and prognosis prediction.
        
        **Key Features:**
        - Multimodal Integration of spatial transcriptomics, pathological images, and clinical text
        - Spatial encoding using graph neural networks to capture cell-molecule spatial relationships
        - Text and Visual encoding for comprehensive analysis
        - Advanced prediction capabilities for tumor classification and prognosis
        """)
    
    # Dashboard page
    else:
        # Sidebar
        with st.sidebar:
            st.markdown(f"### Welcome, {st.session_state.username}")

            # Display user profile information
            if st.session_state.username:
                user_file = f"data/users/{st.session_state.username}.json"
                if os.path.exists(user_file):
                    with open(user_file, "r") as f:
                        user_data = json.load(f)
                    
                    st.markdown(f"**Email:** {user_data['email']}")
                    st.markdown(f"**Last Login:** {user_data.get('last_login', 'First login')}")

            st.markdown("---")
            
            # Navigation
            page = st.radio("Navigation", 
                ["Dashboard", "Data Upload", "Analysis", "Results Visualization", "Settings"])
            
            st.markdown("---")
            
            # Admin section (for admin users)
            if st.session_state.username == "admin":
                st.markdown("### Admin Controls")
                if st.button("User Management"):
                    st.session_state['page'] = "Settings"
                    st.rerun()
            
            # Logout button
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.rerun()
        
        # Main content
        if page == "Dashboard":
            st.markdown('<h1 class="dashboard-header">SPMM Dashboard</h1>', unsafe_allow_html=True)
            st.markdown('<p class="dashboard-subheader">Spatio-Based Pathology MultiModal Model</p>', unsafe_allow_html=True)
            
            # Dashboard metrics - just for demonstration
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="feature-container">', unsafe_allow_html=True)
                st.markdown('### Samples')
                st.markdown('<div class="metric-value">124</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-name">Total</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="feature-container">', unsafe_allow_html=True)
                st.markdown('### Accuracy')
                st.markdown('<div class="metric-value">92.7%</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-name">Overall</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="feature-container">', unsafe_allow_html=True)
                st.markdown('### Processing')
                st.markdown('<div class="metric-value">3</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-name">Jobs Active</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col4:
                st.markdown('<div class="feature-container">', unsafe_allow_html=True)
                st.markdown('### Models')
                st.markdown('<div class="metric-value">5</div>', unsafe_allow_html=True)
                st.markdown('<div class="metric-name">Available</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Recent activity
            st.markdown("### Recent Activity")
            
            # Sample activity data
            activity_data = {
                "Timestamp": ["2025-04-26 09:15:23", "2025-04-26 08:32:17", "2025-04-25 16:45:09", 
                           "2025-04-25 14:20:38", "2025-04-24 11:05:52"],
                "User": ["admin", "researcher2", "admin", "clinician1", "researcher1"],
                "Activity": ["Model prediction", "Data upload", "Settings update", 
                           "Report generation", "Model prediction"],
                "Status": ["Completed", "Completed", "Completed", "Failed", "Completed"]
            }
            
            df_activity = pd.DataFrame(activity_data)
            
            # Add styling
            def highlight_failed(val):
                if val == "Failed":
                    return 'color: red'
                return ''
            
            styled_df = df_activity.style.applymap(highlight_failed, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # System status
            st.markdown("### System Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**CPU Usage**")
                cpu_usage = 32  # Example value
                st.progress(cpu_usage / 100)
                st.write(f"{cpu_usage}%")
            
            with col2:
                st.markdown("**Memory Usage**")
                memory_usage = 45  # Example value
                st.progress(memory_usage / 100)
                st.write(f"{memory_usage}%")
            
            # Model versions
            st.markdown("### Available Model Versions")
            
            model_data = {
                "Version": ["SPMM v1.2.3", "SPMM v1.1.0", "SPMM v1.0.5", "SPMM v1.0.0"],
                "Released": ["2025-03-15", "2024-12-05", "2024-08-22", "2024-05-10"],
                "Accuracy": ["92.7%", "91.3%", "89.5%", "87.2%"],
                "Status": ["Active (Default)", "Available", "Available", "Deprecated"]
            }
            
            df_models = pd.DataFrame(model_data)
            st.dataframe(df_models, use_container_width=True)
        
        elif page == "Data Upload":
            st.markdown('<h1 class="dashboard-header">Data Upload</h1>', unsafe_allow_html=True)
            
            # File upload section
            st.markdown("### Upload Your Data")
            
            st.markdown("#### Spatial Transcriptomics Data")
            spatial_file = st.file_uploader("Upload spatial data (H5AD format)", type=["h5ad"])
            
            st.markdown("#### Pathology Image")
            image_file = st.file_uploader("Upload WSI image", type=["png", "jpg", "tif", "svs"])
            
            st.markdown("#### Clinical Text")
            clinical_file = st.file_uploader("Upload clinical text data", type=["txt"])
            
            # Visualization of uploaded data
            if spatial_file or image_file or clinical_file:
                st.markdown("### Data Preview")
                
                if spatial_file:
                    st.success("Spatial data file uploaded successfully")
                    # In a real application, you would process the actual file
                    # For demo purposes, we show sample data
                    sample_data = load_sample_data()
                    
                    # Visualize first few genes
                    st.markdown("#### Sample Gene Expression")
                    with st.expander("Expand to view sample data"):
                        df_genes = pd.DataFrame(
                            sample_data["gene_expr"][:10, :5], 
                            columns=sample_data["gene_names"][:5]
                        )
                        st.dataframe(df_genes)
                    
                    # Visualize spatial distribution
                    st.markdown("#### Spatial Distribution")
                    buf = visualize_spatial_data(
                        sample_data["coords"],
                        sample_data["cell_types"],
                        "Cell Type Distribution",
                        colormap="tab10"
                    )
                    st.image(buf)
                
                if image_file:
                    st.success("Image file uploaded successfully")
                    # Display the image
                    st.markdown("#### Pathology Image")
                    try:
                        img = Image.open(image_file)
                        st.image(img, caption="Uploaded pathology image")
                    except:
                        st.warning("Could not preview this image format")
                
                if clinical_file:
                    st.success("Clinical text file uploaded successfully")
                    # Read and display text
                    st.markdown("#### Clinical Text Preview")
                    try:
                        clinical_text = clinical_file.read().decode("utf-8")
                        with st.expander("Expand to view clinical text"):
                            st.text_area("Clinical Notes", clinical_text, height=200)
                    except:
                        st.warning("Could not preview text file")
            
            # Form to add metadata
            st.markdown("### Sample Metadata")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sample_id = st.text_input("Sample ID")
                patient_age = st.number_input("Patient Age", min_value=0, max_value=120)
                patient_gender = st.selectbox("Patient Gender", ["Male", "Female", "Other"])
            
            with col2:
                primary_site = st.selectbox("Primary Site", 
                                           ["Breast", "Lung", "Colon", "Prostate", "Brain", "Other"])
                known_diagnosis = st.selectbox("Known Diagnosis", 
                                             ["Unknown", "Normal", "Benign", "Precancerous", "Malignant"])
                sample_date = st.date_input("Sample Collection Date")
            
            # Submit button
            if st.button("Process Data"):
                if spatial_file or (st.session_state.get('demo_mode', False) and sample_id):
                    with st.spinner("Processing data..."):
                        # Simulate processing
                        progress_bar = st.progress(0)
                        for i in range(101):
                            time.sleep(0.02)
                            progress_bar.progress(i)
                        
                        st.success("Data successfully processed and ready for analysis!")
                        st.session_state.data_uploaded = True
                        
                        # Store sample data for demo
                        if not spatial_file:
                            st.session_state.sample_data = load_sample_data()
                else:
                    st.error("Please upload spatial data file or enter a Sample ID in demo mode")
            
            # Demo mode toggle
            st.markdown("---")
            demo_mode = st.checkbox("Enable Demo Mode", value=st.session_state.get('demo_mode', False))
            st.session_state.demo_mode = demo_mode
            
            if demo_mode:
                st.info("Demo mode enabled. You can proceed without uploading actual files.")
        
        elif page == "Analysis":
            st.markdown('<h1 class="dashboard-header">Data Analysis</h1>', unsafe_allow_html=True)
            
            # Check if data is uploaded or demo mode is enabled
            if not st.session_state.get('data_uploaded', False) and not st.session_state.get('demo_mode', False):
                st.warning("Please upload data first or enable demo mode")
                st.button("Go to Data Upload", on_click=lambda: st.session_state.update({"page": "Data Upload"}))
            else:
                # Load sample data for demo
                if st.session_state.get('demo_mode', True) and not st.session_state.get('data_uploaded', False):
                    sample_data = load_sample_data()
                    st.session_state.sample_data = sample_data
                elif not st.session_state.get('sample_data', None):
                    sample_data = load_sample_data()
                    st.session_state.sample_data = sample_data
                else:
                    sample_data = st.session_state.sample_data
                
                # Analysis options
                st.markdown("### Analysis Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_type = st.selectbox("Analysis Type", 
                                               ["Tumor Classification", "Cell Type Identification", 
                                                "Gene Expression Analysis", "Survival Prediction"])
                    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
                
                with col2:
                    model_version = st.selectbox("Model Version", 
                                               ["SPMM v1.2.3 (Default)", "SPMM v1.1.0", "SPMM v1.0.5", "SPMM v1.0.0"])
                    result_resolution = st.selectbox("Result Resolution", ["High", "Medium", "Low"])
                
                # Advanced options
                with st.expander("Advanced Options"):
                    st.markdown("#### Model Parameters")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        spatial_weight = st.slider("Spatial Modality Weight", 0.0, 1.0, 0.33)
                    with col2:
                        text_weight = st.slider("Text Modality Weight", 0.0, 1.0, 0.33)
                    with col3:
                        image_weight = st.slider("Image Modality Weight", 0.0, 1.0, 0.34)
                    
                    # Normalize weights
                    total_weight = spatial_weight + text_weight + image_weight
                    if total_weight > 0:
                        spatial_weight /= total_weight
                        text_weight /= total_weight
                        image_weight /= total_weight
                    
                    st.markdown("#### Gene Selection")
                    
                    # In a real application, you would load the actual gene names
                    gene_options = sample_data["gene_names"]
                    selected_genes = st.multiselect("Select specific genes for analysis", 
                                                  options=gene_options, 
                                                  default=gene_options[:5])
                
                # Run analysis button
                if st.button("Run Analysis"):
                    with st.spinner("Running analysis..."):
                        # Simulate processing
                        progress_bar = st.progress(0)
                        for i in range(101):
                            time.sleep(0.03)
                            progress_bar.progress(i)
                        
                        # Generate mock predictions
                        predictions = predict_with_model(sample_data["gene_expr"], sample_data["coords"])
                        st.session_state.predictions = predictions
                        
                        st.success("Analysis completed successfully!")
                
                # Display results
                if st.session_state.get('predictions', None):
                    st.markdown("### Analysis Results")
                    
                    predictions = st.session_state.predictions
                    
                    # Prediction visualization
                    class_names = ["Normal", "Benign", "Precancerous", "Malignant"]
                    
                    # Overview metrics
                    st.markdown("#### Prediction Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Count predictions by class
                    pred_counts = np.bincount(predictions["pred_classes"], minlength=4)
                    pred_percentages = pred_counts / pred_counts.sum() * 100
                    
                    with col1:
                        st.metric("Normal", f"{pred_counts[0]} spots", f"{pred_percentages[0]:.1f}%")
                    with col2:
                        st.metric("Benign", f"{pred_counts[1]} spots", f"{pred_percentages[1]:.1f}%")
                    with col3:
                        st.metric("Precancerous", f"{pred_counts[2]} spots", f"{pred_percentages[2]:.1f}%")
                    with col4:
                        st.metric("Malignant", f"{pred_counts[3]} spots", f"{pred_percentages[3]:.1f}%")
                    
                    # Spatial visualization
                    st.markdown("#### Spatial Prediction Distribution")
                    
                    buf = visualize_spatial_data(
                        sample_data["coords"],
                        predictions["pred_classes"],
                        "Spatial Prediction Distribution",
                        colormap="viridis",
                        size=50
                    )
                    st.image(buf)
                    
                    # Confidence heatmap
                    st.markdown("#### Prediction Confidence")
                    
                    # Calculate max probability for each spot as confidence
                    confidence = np.max(predictions["classification"], axis=1)
                    
                    buf = visualize_spatial_data(
                        sample_data["coords"],
                        confidence,
                        "Prediction Confidence Distribution",
                        colormap="plasma",
                        size=50
                    )
                    st.image(buf)
                    
                    # Feature importance
                    st.markdown("#### Feature Importance")
                    
                    # Generate mock feature importance
                    feature_importance = np.abs(np.random.randn(10))
                    feature_importance = feature_importance / feature_importance.sum()
                    feature_names = ["Feature " + str(i+1) for i in range(10)]
                    
                    # Sort by importance
                    idx = np.argsort(feature_importance)[::-1]
                    feature_importance = feature_importance[idx]
                    feature_names = [feature_names[i] for i in idx]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(feature_names, feature_importance, color='skyblue')
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close(fig)
                    
                    st.image(buf)
        
        elif page == "Results Visualization":
            st.markdown('<h1 class="dashboard-header">Results Visualization</h1>', unsafe_allow_html=True)
            
            # Check if analysis has been run
            if not st.session_state.get('predictions', None) and not st.session_state.get('demo_mode', False):
                st.warning("Please run analysis first")
                st.button("Go to Analysis", on_click=lambda: st.session_state.update({"page": "Analysis"}))
            else:
                # If demo mode but no predictions, generate them
                if st.session_state.get('demo_mode', False) and not st.session_state.get('predictions', None):
                    if not st.session_state.get('sample_data', None):
                        sample_data = load_sample_data()
                        st.session_state.sample_data = sample_data
                    else:
                        sample_data = st.session_state.sample_data
                    
                    predictions = predict_with_model(sample_data["gene_expr"], sample_data["coords"])
                    st.session_state.predictions = predictions
                
                # Load sample data and predictions
                sample_data = st.session_state.sample_data
                predictions = st.session_state.predictions
                
                # Visualization settings
                st.markdown("### Visualization Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    viz_type = st.selectbox("Visualization Type", 
                                         ["Spatial Heatmap", "Cell Type Distribution", 
                                          "Gene Expression", "Feature Space", "Prediction Distribution"])
                    color_scheme = st.selectbox("Color Scheme", ["viridis", "plasma", "inferno", "magma", "cividis"])
                
                with col2:
                    resolution = st.selectbox("Resolution", ["High", "Medium", "Low"])
                    interactive = st.checkbox("Interactive Visualization", value=True)
                
                # Visualization options based on selected type
                if viz_type == "Gene Expression":
                    st.markdown("#### Gene Selection")
                    
                    # In a real application, you would load the actual gene names
                    gene_options = sample_data["gene_names"]
                    selected_gene = st.selectbox("Select gene to visualize", 
                                              options=gene_options)
                elif viz_type == "Cell Type Distribution":
                    st.markdown("#### Cell Type Options")
                    
                    cell_types = ["All Types"] + [f"Type {i}" for i in range(5)]
                    selected_cell_types = st.multiselect("Select cell types to display", 
                                                      options=cell_types, 
                                                      default=["All Types"])
                
                # Render visualization
                st.markdown("### Visualization Results")
                
                if viz_type == "Spatial Heatmap":
                    st.markdown("#### Prediction Spatial Heatmap")
                    
                    # Visualize prediction classes
                    buf = visualize_spatial_data(
                        sample_data["coords"],
                        predictions["pred_classes"],
                        "Tumor Classification Prediction",
                        colormap=color_scheme,
                        size=50 if resolution == "Low" else (100 if resolution == "Medium" else 150)
                    )
                    st.image(buf)
                    
                    # Add legend
                    st.markdown("**Legend:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**0:** Normal")
                    with col2:
                        st.markdown("**1:** Benign")
                    with col3:
                        st.markdown("**2:** Precancerous")
                    with col4:
                        st.markdown("**3:** Malignant")
                
                elif viz_type == "Cell Type Distribution":
                    st.markdown("#### Cell Type Distribution")
                    
                    # Filter cell types if specific types are selected
                    if "All Types" not in selected_cell_types:
                        mask = np.zeros_like(sample_data["cell_types"], dtype=bool)
                        for cell_type in selected_cell_types:
                            type_id = int(cell_type.split(" ")[1])
                            mask |= (sample_data["cell_types"] == type_id)
                        
                        filtered_coords = sample_data["coords"][mask]
                        filtered_types = sample_data["cell_types"][mask]
                        
                        buf = visualize_spatial_data(
                            filtered_coords,
                            filtered_types,
                            "Selected Cell Type Distribution",
                            colormap=color_scheme,
                            size=50 if resolution == "Low" else (100 if resolution == "Medium" else 150)
                        )
                    else:
                        buf = visualize_spatial_data(
                            sample_data["coords"],
                            sample_data["cell_types"],
                            "Cell Type Distribution",
                            colormap=color_scheme,
                            size=50 if resolution == "Low" else (100 if resolution == "Medium" else 150)
                        )
                    
                    st.image(buf)
                
                elif viz_type == "Gene Expression":
                    st.markdown(f"#### Expression of {selected_gene}")
                    
                    # Get gene index
                    gene_idx = sample_data["gene_names"].index(selected_gene)
                    
                    # Visualize gene expression
                    buf = visualize_spatial_data(
                        sample_data["coords"],
                        sample_data["gene_expr"][:, gene_idx],
                        f"Spatial Expression of {selected_gene}",
                        colormap=color_scheme,
                        size=50 if resolution == "Low" else (100 if resolution == "Medium" else 150)
                    )
                    st.image(buf)
                
                elif viz_type == "Feature Space":
                    st.markdown("#### Feature Space Visualization")
                    
                    # Generate mock feature space visualization
                    # In a real application, you would use actual features from the model
                    
                    # Use PCA to reduce feature dimensions for visualization
                    from sklearn.decomposition import PCA
                    
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(predictions["features"])
                    
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                        c=predictions["pred_classes"], 
                                        cmap=color_scheme, 
                                        s=50 if resolution == "Low" else (100 if resolution == "Medium" else 150),
                                        alpha=0.8)
                    plt.colorbar(scatter, label='Predicted Class')
                    plt.title('Feature Space (PCA)')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    st.image(buf)
                    
                    # Add legend
                    st.markdown("**Legend:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**0:** Normal")
                    with col2:
                        st.markdown("**1:** Benign")
                    with col3:
                        st.markdown("**2:** Precancerous")
                    with col4:
                        st.markdown("**3:** Malignant")
                
                elif viz_type == "Prediction Distribution":
                    st.markdown("#### Prediction Distribution")
                    
                    # Create a histogram of prediction classes
                    class_names = ["Normal", "Benign", "Precancerous", "Malignant"]
                    class_counts = np.bincount(predictions["pred_classes"], minlength=4)
                    
                    plt.figure(figsize=(10, 6))
                    bars = plt.bar(class_names, class_counts, color=['green', 'blue', 'orange', 'red'])
                    plt.title('Prediction Distribution')
                    plt.ylabel('Number of Spots')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{height}', ha='center', va='bottom')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    st.image(buf)
                    
                    # Add pie chart
                    plt.figure(figsize=(8, 8))
                    plt.pie(class_counts, labels=class_names, autopct='%1.1f%%', 
                          startangle=90, colors=['green', 'blue', 'orange', 'red'])
                    plt.axis('equal')
                    plt.title('Prediction Distribution (%)')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    plt.close()
                    
                    col1, col2 = st.columns(2)
                    
                    with col2:
                        st.image(buf)
                
                # Export options
                st.markdown("### Export Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        "Download as PNG",
                        data=buf,
                        file_name=f"spmm_{viz_type.lower().replace(' ', '_')}.png",
                        mime="image/png"
                    )
                
                with col2:
                    st.download_button(
                        "Download Data (CSV)",
                        data=pd.DataFrame({
                            "x_coord": sample_data["coords"][:, 0],
                            "y_coord": sample_data["coords"][:, 1],
                            "prediction": predictions["pred_classes"]
                        }).to_csv(index=False),
                        file_name="spmm_predictions.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    st.download_button(
                        "Full Report (PDF)",
                        data=b"Sample report data - this would be a PDF in a real application",
                        file_name="spmm_report.pdf",
                        mime="application/pdf"
                    )
        
        elif page == "Settings":
            st.markdown('<h1 class="dashboard-header">Settings</h1>', unsafe_allow_html=True)
            
            # General settings
            st.markdown("### General Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                dark_mode = st.checkbox("Dark Mode", value=False)
                notifications = st.checkbox("Enable Notifications", value=True)
            
            with col2:
                language = st.selectbox("Language", ["English", "Chinese", "French", "German", "Japanese"])
                auto_save = st.checkbox("Auto-save Results", value=True)
            
            # Model settings
            st.markdown("### Model Settings")
            
            default_model = st.selectbox("Default Model Version", 
                                       ["SPMM v1.2.3", "SPMM v1.1.0", "SPMM v1.0.5", "SPMM v1.0.0"])
            
            col1, col2 = st.columns(2)
            
            with col1:
                gpu_acceleration = st.checkbox("Enable GPU Acceleration", value=True)
                cache_results = st.checkbox("Cache Prediction Results", value=True)
            
            with col2:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=16)
                precision = st.selectbox("Computation Precision", ["FP32", "FP16", "INT8"])
            
            # User management
            if st.session_state.username == "admin":
                st.markdown("### User Management (Admin Only)")
                
                with st.expander("Add New User"):
                    new_username = st.text_input("New Username")
                    new_password = st.text_input("New Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    
                    if st.button("Add User"):
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif not new_username or not new_password:
                            st.error("Username and password cannot be empty")
                        else:
                            add_user(new_username, new_password)
                            st.success(f"User {new_username} added successfully")
                
                # Display existing users
                with open("data/credentials.json", "r") as f:
                    credentials = json.load(f)
                
                user_list = list(credentials.keys())
                
                st.markdown("#### Existing Users")
                st.dataframe(pd.DataFrame({"Username": user_list}))
            
            # Save settings button
            if st.button("Save Settings"):
                st.success("Settings saved successfully")
                
                # In a real application, you would save these settings to a file or database
                # For demo purposes, we just show a success message

# Run the app
if __name__ == "__main__":
    main()