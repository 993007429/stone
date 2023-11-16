import functools
from typing import List, Type

from apiflask import abort


class BasePermission:
    def has_permission(self, view_func):
        return True


def check_permissions(permissions: List[BasePermission], view_func) -> bool:
    for permission in permissions:
        if permission.has_permission(view_func):
            return True
    return False


def permission_required(permission_classes: List[Type[BasePermission]]):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            permissions = [permission() for permission in permission_classes]
            if not check_permissions(permissions, func):
                return {'code': 403, 'message': 'Permission denied'}
            return func(*args, **kwargs)

        return wrapper

    return decorate
