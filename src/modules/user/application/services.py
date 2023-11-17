from typing import List

from src.app.request_context import request_context
from src.modules.user.domain.services import UserDomainService
from src.seedwork.application.responses import AppResponse


class UserService(object):

    def __init__(self, domain_service: UserDomainService):
        self.domain_service = domain_service

    def create_user(self, **kwargs) -> AppResponse[dict]:
        new_user, message = self.domain_service.create_user(**kwargs)
        if not new_user:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data=new_user.dict())

    def get_users(self) -> AppResponse[List[dict]]:
        users = self.domain_service.get_users()
        return AppResponse(message='get users success', data=[user.dict() for user in users])

    def get_user(self, userid: int) -> AppResponse[dict]:
        user, message = self.domain_service.get_user(userid)
        if not user:
            return AppResponse(message=message)
        return AppResponse(message=message, data=user.dict())

    def update_user(self, userid: int) -> AppResponse[dict]:
        user, message = self.domain_service.update_user(userid)
        if not user:
            return AppResponse(message=message)
        return AppResponse(message=message, data=user.dict())

    def login(self, **kwargs) -> AppResponse[dict]:
        login_user, message = self.domain_service.login(**kwargs)
        if not login_user:
            return AppResponse(message=message)
        request_context.current_user = login_user
        return AppResponse(message=message, data=login_user.dict())

    def delete_user(self, userid: int) -> AppResponse[dict]:
        user, message = self.domain_service.delete_user(userid)
        if not user:
            return AppResponse(message=message)
        return AppResponse(message=message, data=user.dict())