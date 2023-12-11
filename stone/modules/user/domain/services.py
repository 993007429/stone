from typing import Optional, List, Tuple

from stone.app.request_context import request_context
from stone.modules.user.domain.entities import UserEntity
from stone.modules.user.domain.value_objects import LoginUser
from stone.modules.user.infrastructure.repositories import SQLAlchemyUserRepository
from stone.modules.user.utils.auth import verify_password, hash_password, get_token_for_user


class UserDomainService(object):

    def __init__(self, user_repository: SQLAlchemyUserRepository):
        self.user_repository = user_repository

    def create_user(self, **kwargs) -> Tuple[Optional[UserEntity], str]:
        kwargs['password_hash'] = hash_password(kwargs['password'])
        del kwargs['password']

        if not kwargs.get('creator'):
            current_user = request_context.current_user
            kwargs['creator'] = current_user.username
        user_to_save = UserEntity.parse_obj(kwargs)
        new_user = self.user_repository.save(user_to_save)
        if not new_user:
            return None, 'Create user failed'
        return new_user, 'Create user succeed'

    def get_users(self, **kwargs) -> Tuple[List[UserEntity], dict, str]:
        page = kwargs['page']
        per_page = kwargs['per_page']
        names_to_exclude = ['sa']
        users, pagination = self.user_repository.get_users(page, per_page, names_to_exclude)
        return users, pagination, 'Get users succeed'

    def get_user(self, userid: int) -> Tuple[Optional[UserEntity], str]:
        user = self.user_repository.get(userid)
        if not user:
            return None, 'No user'
        return user, 'Get user success'

    def update_user(self, **kwargs) -> Tuple[Optional[UserEntity], str]:
        user_id = kwargs['user_id']
        user_data = kwargs['user_data']
        user_data['password_hash'] = hash_password(user_data['password'])
        del user_data['password']

        updated_count = self.user_repository.update(user_id, user_data)
        if not updated_count:
            return None, 'Update user failed'
        user = self.user_repository.get(user_id)
        return user, 'Update user succeed'

    def delete_user(self, userid: int) -> Tuple[int, str]:
        deleted_count = self.user_repository.delete(userid)
        if not deleted_count:
            return deleted_count, 'No user deleted'
        return deleted_count, 'Delete user succeed'

    def login(self, **kwargs) -> Tuple[Optional[LoginUser], str]:
        user = self.user_repository.get_user_by_name(kwargs['username'])
        if not user:
            return None, 'No user'

        if not verify_password(kwargs['password'], user.password_hash):
            return None, 'Wrong password'

        token = get_token_for_user(user)
        return LoginUser(userid=user.id,
                         username=user.username,
                         role=user.role,
                         token=token), 'Login success'
