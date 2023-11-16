from src.app.permission import BasePermission
from src.app.request_context import request_context
from src.modules.user.domain.value_objects import RoleType


class IsAdmin(BasePermission):

    def has_permission(self, view_func):
        current_user = request_context.current_user
        return current_user.role == RoleType.admin.value
