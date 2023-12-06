from stone.app.permission import BasePermission
from stone.app.request_context import request_context
from stone.modules.user.domain.value_objects import RoleType


class IsAdmin(BasePermission):

    def has_permission(self, view_func, *args, **kwargs):
        current_user = request_context.current_user
        return current_user.role == RoleType.admin.value


class DeleteAnalysisPermission(BasePermission):

    def has_permission(self, view_func, *args, **kwargs):
        from stone.app.service_factory import AppServiceFactory
        from stone.modules.ai.domain.services import AiDomainService
        analysis_id = kwargs['analysis_id']
        _self: AiDomainService = AppServiceFactory.ai_service.domain_service
        analysis = _self.repository.get_analysis_by_pk(analysis_id)
        if not analysis:
            return False
        # if analysis.userid != request_context.current_user.userid:
        if analysis.userid != 1:
            return False
        return True
