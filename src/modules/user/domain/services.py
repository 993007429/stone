from typing import Optional, List

from src.modules.user.domain.entities import UserEntity
from src.modules.user.domain.value_objects import LoginInfo
from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository
from src.modules.user.utils.auth import verify_password, hash_password, get_token_for_user


class UserDomainService(object):

    def __init__(self, repository: SQLAlchemyUserRepository):
        self.repository = repository

    def create_user(self, **kwargs) -> Optional[UserEntity]:
        kwargs['password_hash'] = hash_password(kwargs['password'])
        del kwargs['password']

        new_user = UserEntity(**kwargs)
        if self.repository.save(new_user):
            return new_user
        return None

    def get_users(self) -> List[Optional[UserEntity]]:
        users = self.repository.gets()
        return [user for user in users]

    def login(self, **kwargs) -> Optional[LoginInfo]:
        user = self.repository.get(kwargs['username'])
        if not user:
            return None

        if not verify_password(kwargs['password'], user.password_hash):
            return None

        token = get_token_for_user(user)
        return LoginInfo(userid=user.id,
                         username=user.username,
                         role=user.role,
                         token=token)
