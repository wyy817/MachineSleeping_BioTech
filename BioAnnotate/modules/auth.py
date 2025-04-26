"""
Authentication module for user management.
"""

import streamlit as st
import json
import os
import hashlib
import time
import uuid
from datetime import datetime

# Import configuration
import config

def initialize_user_db():
    """Initialize the user database if it doesn't exist."""
    if not os.path.exists(config.USER_DB_PATH):
        with open(config.USER_DB_PATH, 'w') as f:
            json.dump({}, f)

def hash_password(password):
    """Create a hash of the password for secure storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_users():
    """Get all users from the database."""
    initialize_user_db()
    try:
        with open(config.USER_DB_PATH, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is empty or invalid JSON
        return {}

def save_users(users):
    """Save users to the database."""
    with open(config.USER_DB_PATH, 'w') as f:
        json.dump(users, f, indent=4)

def create_account():
    """Create a new user account."""
    st.subheader("Create an Account")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        submitted = st.form_submit_button("Create Account")
        
        if submitted:
            if not username or not password:
                st.error("Username and password are required")
                return False
                
            if password != confirm_password:
                st.error("Passwords do not match")
                return False
                
            # Check if username already exists
            users = get_users()
            if username in users:
                st.error(f"Username '{username}' already exists")
                return False
                
            # Create new user
            users[username] = {
                "password_hash": hash_password(password),
                "created_at": datetime.now().isoformat(),
                "user_id": str(uuid.uuid4())
            }
            
            save_users(users)
            
            # Set session state
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_id = users[username]["user_id"]
            
            st.success("Account created successfully!")
            st.info("You are now logged in.")
            
            # Create user directories
            user_upload_dir = os.path.join(config.USER_UPLOADS_DIR, username)
            user_annotations_dir = os.path.join(config.ANNOTATIONS_DIR, username)
            
            os.makedirs(user_upload_dir, exist_ok=True)
            os.makedirs(user_annotations_dir, exist_ok=True)
            
            return True
            
    return False

def login():
    """Log in a user."""
    st.subheader("Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if not username or not password:
                st.error("Username and password are required")
                return False
                
            users = get_users()
            
            if username not in users:
                st.error("Invalid username or password")
                return False
                
            stored_hash = users[username]["password_hash"]
            input_hash = hash_password(password)
            
            if stored_hash != input_hash:
                st.error("Invalid username or password")
                return False
                
            # Set session state
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_id = users[username]["user_id"]
            
            st.success("Login successful!")
            
            # Update last login time
            users[username]["last_login"] = datetime.now().isoformat()
            save_users(users)
            
            return True
            
    return False

def logout():
    """Log out the current user."""
    # Clear session state
    for key in ["authenticated", "username", "user_id"]:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("You have been logged out.")
    return True

def is_authenticated():
    """Check if a user is authenticated."""
    return st.session_state.get("authenticated", False)

def get_current_user():
    """Get the current authenticated user."""
    if not is_authenticated():
        return None
    
    return {
        "username": st.session_state.username,
        "user_id": st.session_state.user_id
    }

def require_auth(func):
    """Decorator to require authentication for a function."""
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            st.warning("Please log in to access this feature.")
            login()
            return None
        return func(*args, **kwargs)
    
    return wrapper