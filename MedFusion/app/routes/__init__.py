# Register all blueprints
from flask import Blueprint

def register_routes(app):
    from app.routes import auth, personal, doctor, warehouse, admin
    app.register_blueprint(auth.bp)
    app.register_blueprint(personal.bp)
    app.register_blueprint(doctor.bp)
    app.register_blueprint(warehouse.bp)
    app.register_blueprint(admin.bp)
