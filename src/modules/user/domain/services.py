from typing import Optional, List

from src.modules.user.domain.entities import UserEntity
from src.modules.user.domain.repositories import UserRepository
from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


class UserDomainService(object):

    def __init__(self, repository: SQLAlchemyUserRepository):
        self.repository = repository

    def create_user(self, **kwargs) -> Optional[UserEntity]:
        new_user = UserEntity.create_user(**kwargs)
        if self.repository.save(new_user):
            return new_user
        return None

    def get_users(self) -> List[Optional[UserEntity]]:
        users = self.repository.gets()
        return [user for user in users]

