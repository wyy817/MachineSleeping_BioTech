from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from app.forms import RegisterForm, LoginForm
from app.models import read_users, write_user, User
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint("auth", __name__)


@bp.route("/")
def home():
    return redirect(url_for("auth.login"))


@bp.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_users = read_users()
        if any(u["email"] == form.email.data for u in existing_users):
            flash("Email already exists", "danger")
            return redirect(url_for("auth.register"))

        # 生成哈希密码
        password_hash = generate_password_hash(form.password.data)

        # 创建新用户
        new_user = {
            "id": len(existing_users) + 1,
            "username": form.username.data,
            "email": form.email.data,
            "roles": "user",
            "password_hash": password_hash
        }

        # 写入CSV
        write_user(new_user)
        flash("Successful registration, please login!", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html", form=form)


@bp.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        users = read_users()
        user_dict = next((u for u in users if u["email"] == form.email.data), None)

        if user_dict and check_password_hash(user_dict["password_hash"], form.password.data):
            user = User(user_dict)
            login_user(user)
            flash("Login successfully", "success")
            return redirect(url_for("dashboard.index"))
        else:
            flash("Email or password is wrong.", "danger")
    return render_template("auth/login.html", form=form)


@bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logout", "info")
    return redirect(url_for("auth.login"))
