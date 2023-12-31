from typing import List

from stone.app.request_context import request_context
from stone.modules.user.domain.services import UserDomainService
from stone.seedwork.application.responses import AppResponse


class UserService(object):

    def __init__(self, domain_service: UserDomainService):
        self.domain_service = domain_service

    def create_user(self, **kwargs) -> AppResponse[dict]:
        new_user, message = self.domain_service.create_user(**kwargs)
        if not new_user:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'user': new_user.dict()})

    def get_users(self, **kwargs) -> AppResponse[List[dict]]:
        users, pagination, message = self.domain_service.get_users(**kwargs)
        return AppResponse(message=message, data={'users': [user.dict() for user in users]})

    def get_user(self, userid: int) -> AppResponse[dict]:
        user, message = self.domain_service.get_user(userid)
        if not user:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'user': user.dict()})

    def update_user(self, **kwargs) -> AppResponse[dict]:
        user, message = self.domain_service.update_user(**kwargs)
        if not user:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'user': user.dict()})

    def login(self, **kwargs) -> AppResponse[dict]:
        login_user, message = self.domain_service.login(**kwargs)
        if not login_user:
            return AppResponse(err_code=1, message=message)
        request_context.token = login_user.token
        request_context.current_user = login_user
        return AppResponse(message=message, data={'login': login_user.dict()})

    def delete_user(self, userid: int) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_user(userid)
        return AppResponse(message=message, data={'affected_count': deleted_count})
