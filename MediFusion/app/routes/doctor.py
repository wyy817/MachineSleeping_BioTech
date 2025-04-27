from flask import Blueprint, render_template, flash, redirect, url_for
from flask_login import login_required, current_user

bp = Blueprint("doctor", __name__, url_prefix="/doctor")


@bp.route("/model")
@login_required
def model():
    if "Doctor" not in current_user.roles:
        flash("Sorry, only doctor can reach this module.", "warning")
        return redirect(url_for("dashboard.index"))
