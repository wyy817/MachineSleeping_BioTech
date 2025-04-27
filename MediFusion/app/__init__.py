from flask import Flask
from flask_login import LoginManager

from app.routes import dashboard

# Initialize the login manager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # Secret key for session encryption and CSRF protection
    app.config['SECRET_KEY'] = 'your_default_secret_key'

    # Initialize LoginManager
    login_manager.init_app(app)

    from app.models import load_user

    # Register blueprints (routes)
    from app.routes import auth, personal, doctor, warehouse, admin
    app.register_blueprint(auth.bp)
    app.register_blueprint(personal.bp)
    app.register_blueprint(doctor.bp)
    app.register_blueprint(warehouse.bp)
    app.register_blueprint(admin.bp)
    app.register_blueprint(dashboard.bp)

    return app
