from stone.app.permission import BasePermission
from stone.app.request_context import request_context
from stone.modules.user.domain.value_objects import RoleType


class IsAdmin(BasePermission):

    def has_permission(self, view_func):
        current_user = request_context.current_user
        return current_user.role == RoleType.admin.value
