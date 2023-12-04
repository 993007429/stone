from typing import Optional, List, Tuple

from stone.modules.user.domain.entities import UserEntity
from stone.modules.user.domain.value_objects import LoginUser
from stone.modules.user.infrastructure.repositories import SQLAlchemyUserRepository
from stone.modules.user.utils.auth import verify_password, hash_password, get_token_for_user


class UserDomainService(object):

    def __init__(self, repository: SQLAlchemyUserRepository):
        self.repository = repository

    def create_user(self, **kwargs) -> Tuple[Optional[UserEntity], str]:
        kwargs['password_hash'] = hash_password(kwargs['password'])
        del kwargs['password']

        new_user = UserEntity.parse_obj(kwargs)
        success, message = self.repository.save(new_user)
        if success:
            user = self.repository.get_user_by_name(kwargs['username'])
            return user, message
        return None, message

    def get_users(self) -> List[Optional[UserEntity]]:
        users = self.repository.gets()
        return [user for user in users]

    def get_user(self, userid: int) -> Tuple[Optional[UserEntity], str]:
        user = self.repository.get_user_by_pk(userid)
        if not user:
            return None, 'no user'
        return user, 'get user success'

    def update_user(self, userid: int, **kwargs) -> Tuple[Optional[UserEntity], str]:
        user = self.repository.get_user_by_pk(userid)
        if not user:
            return None, 'no user'

        kwargs['password_hash'] = hash_password(kwargs['password'])
        del kwargs['password']

        new_user = user.copy(update=kwargs)

        success, message = self.repository.update(userid, new_user)
        if success:
            user = self.repository.get_user_by_pk(userid)
            return user, message
        return None, message

    def delete_user(self, userid: int) -> Tuple[Optional[UserEntity], str]:
        user = self.repository.get_user_by_pk(userid)
        if not user:
            return None, 'no user'

        success, message = self.repository.delete_user_by_pk(userid)
        return user, message

    def login(self, **kwargs) -> Tuple[Optional[LoginUser], str]:
        user = self.repository.get_user_by_name(kwargs['username'])
        if not user:
            return None, 'No user'

        if not verify_password(kwargs['password'], user.password_hash):
            return None, 'Wrong password'

        token = get_token_for_user(user)
        return LoginUser(userid=user.id,
                         username=user.username,
                         role=user.role,
                         token=token), 'Login success'
