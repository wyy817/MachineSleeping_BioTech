import csv
from datetime import datetime
from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app.forms import DatasetUploadForm
from app.models import read_datasets, write_dataset

bp = Blueprint("warehouse", __name__, url_prefix="/warehouse")


@bp.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    form = DatasetUploadForm()
    if form.validate_on_submit():
        datasets = read_datasets()
        existing_ids = [d['id'] for d in datasets] if datasets else [0]
        new_id = max(existing_ids) + 1

        dataset = {
            'id': new_id,
            'name': form.name.data,
            'description': form.description.data,
            'tags': form.tags.data,
            'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'approval_status': 'pending',
            'approval_comment': '',
            'uploader': current_user.id
        }
        write_dataset(dataset, mode='append')
        flash("Dataset uploaded successfully, waiting for approval.", "success")
        return redirect(url_for("warehouse.my_data"))
    return render_template("warehouse/upload.html", form=form)


def read_datasets():
    datasets = []
    try:
        with open('datasets.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                row['id'] = int(row['id'])
                row['uploader'] = int(row['uploader'])
                datasets.append(row)
    except FileNotFoundError:
        pass
    return datasets


def write_dataset(data, mode='append'):
    fieldnames = ['id', 'name', 'description', 'tags', 'upload_time',
                  'approval_status', 'approval_comment', 'uploader']

    if mode == 'overwrite':
        with open('datasets.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for d in data:
                d['id'] = str(d['id'])
                d['uploader'] = str(d['uploader'])
            writer.writerows(data)
    elif mode == 'append':
        with open('datasets.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if file.tell() == 0:
                writer.writeheader()
            data['id'] = str(data['id'])
            data['uploader'] = str(data['uploader'])
            writer.writerow(data)


@bp.route("/my-data")
@login_required
def my_data():
    datasets = read_datasets()
    if "Warehouse Manager" in current_user.roles:
        filtered = datasets
    else:
        filtered = [
            d for d in datasets
            if d['approval_status'] == 'approved'
            or str(d['uploader']) == str(current_user.id)
        ]
    return render_template("warehouse/my_data.html", datasets=filtered)


@bp.route("/set-tags/<int:dataset_id>", methods=["GET", "POST"])
@login_required
def set_tags(dataset_id):
    datasets = read_datasets()
    dataset = next((d for d in datasets if d['id'] == dataset_id), None)
    if dataset and dataset['uploader'] == current_user.id:
        if request.method == "POST":
            dataset['tags'] = request.form['tags']
            write_dataset(datasets, mode='overwrite')
            flash("Tags updated successfully.", "success")
            return redirect(url_for("warehouse.my_data"))
        return render_template("warehouse/set_tags.html", dataset=dataset)
    flash("You can only edit tags for your own datasets.", "danger")
    return redirect(url_for("warehouse.my_data"))


@bp.route("/delete/<int:dataset_id>")
@login_required
def delete(dataset_id):
    datasets = read_datasets()
    dataset = next((d for d in datasets if d['id'] == dataset_id), None)

    if not dataset:
        flash("Dataset not found.", "danger")
    else:
        if (str(dataset['uploader']) == str(current_user.id)
                or "Warehouse Manager" in current_user.roles):
            new_datasets = [d for d in datasets if d['id'] != dataset_id]
            write_dataset(new_datasets, mode='overwrite')
            flash("Dataset deleted successfully.", "success")
        else:
            flash("You can only delete your own datasets.", "danger")
    return redirect(url_for("warehouse.my_data"))
