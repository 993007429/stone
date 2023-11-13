from typing import List

from src.modules.user.domain.services import UserDomainService
from src.seedwork.application.responses import AppResponse


class UserService(object):

    def __init__(self, domain_service: UserDomainService):
        # super(UserService, self).__init__()
        self.domain_service = domain_service

    def create_user(self, username: str, password: str) -> AppResponse[dict]:
        new_user = self.domain_service.create_user(username, password)
        if not new_user:
            AppResponse(err_code=1, message='create user failed')
        return AppResponse(message='create user success', data=new_user.dict())

    def get_users(self) -> AppResponse[List[dict]]:
        users = self.domain_service.get_users()
        return AppResponse(message='create user success', data={'users': [user.dict for user in users]})
