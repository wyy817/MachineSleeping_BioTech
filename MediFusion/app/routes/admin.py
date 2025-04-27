from flask import Blueprint, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.models import read_datasets, write_dataset

bp = Blueprint("admin", __name__, url_prefix="/admin")


def has_admin_access(user):
    roles = user.roles.split(',')
    return "Warehouse Manager" in roles


@bp.route("/approve/<int:dataset_id>")
@login_required
def approve(dataset_id):
    # Warehouse Manager can access
    if not has_admin_access(current_user):
        flash("Can not approve dataset", "danger")
        return redirect(url_for("dashboard.index"))

    datasets = read_datasets()
    dataset = next((d for d in datasets if int(d['id']) == dataset_id), None)
    if dataset:
        dataset['approval_status'] = 'approved'
        write_dataset(dataset, mode='update')
        flash("The dataset has been approved", "success")
    return redirect(url_for("admin.manage_roles"))
