def has_role(user, role):
    return user.is_authenticated and role in user.roles


def has_any_role(user, role_list):
    return user.is_authenticated and any(role in user.roles for role in role_list)


def is_admin(user):
    return has_role(user, "Admin") or has_role(user, "Warehouse Manager")


def is_leader(user):
    return has_role(user, "Warehouse Manager")
