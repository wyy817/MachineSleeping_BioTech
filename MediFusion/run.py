from app import create_app

# Choose the appropriate config for your environment
app = create_app()

if __name__ == "__main__":
    # Start the Flask server
    app.run(debug=True)  # Set debug=False for production
