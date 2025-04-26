import csv
import os
from flask_login import UserMixin  # 新增导入
from app import login_manager      # 导入 login_manager

# Paths to store CSV files
USER_CSV = 'users.csv'
DATASET_CSV = 'datasets.csv'


# Helper functions to read from CSV files
def read_users():
    users = []
    if os.path.exists(USER_CSV):
        with open(USER_CSV, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                users.append(row)
    return users


def read_datasets():
    datasets = []
    if os.path.exists(DATASET_CSV):
        with open(DATASET_CSV, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                datasets.append(row)
    return datasets


# Simulate adding users and datasets by writing to CSV files
def write_user(user, mode='append'):
    """写入用户数据
    :param mode: 'append' 追加新用户 | 'update' 更新现有用户
    """
    if mode == 'append':
        with open(USER_CSV, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["id", "username", "email", "roles", "password_hash"])
            writer.writerow(user)
    elif mode == 'update':
        users = read_users()
        # 找到并更新用户
        for i, u in enumerate(users):
            if u['id'] == user['id']:
                users[i] = user
                break
        # 重写整个文件
        with open(USER_CSV, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=["id", "username", "email", "roles", "password_hash"])
            writer.writeheader()
            writer.writerows(users)


def write_dataset(dataset):
    with open(DATASET_CSV, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "name", "description", "tags", "user_id", "approval_status", "approval_comment"])
        writer.writerow(dataset)


# 定义 User 类（继承 UserMixin）
class User(UserMixin):
    def __init__(self, user_dict):
        self.id = str(user_dict['id'])          # 必须转为字符串
        self.username = user_dict['username']
        self.email = user_dict['email']
        self.roles = user_dict["roles"].split(',')
        self.password_hash = user_dict['password_hash']


# 用户加载器
@login_manager.user_loader
def load_user(user_id):
    users = read_users()
    user_dict = next((u for u in users if str(u['id']) == user_id), None)
    if user_dict:
        return User(user_dict)
    return None


# Dataset 类保持不变
class Dataset:
    def __init__(self, id, name, description, tags, user_id, approval_status, approval_comment):
        self.id = id
        self.name = name
        self.description = description
        self.tags = tags
        self.user_id = user_id
        self.approval_status = approval_status
        self.approval_comment = approval_comment
