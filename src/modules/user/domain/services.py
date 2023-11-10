from src.modules.user.domain.entities import UserEntity
from src.modules.user.domain.repositories import UserRepository
from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository


class UserDomainService(object):

    def __init__(self, repository: SQLAlchemyUserRepository):
        # super(UserDomainService, self).__init__()
        self.repository = repository

    def create_user(self, username: str, password: str):
        new_user = UserEntity(
            username=username,
            password=password
        )

        flag = self.repository.save(new_user)
        if flag:
            return '', new_user
        return 'create user failed', None
