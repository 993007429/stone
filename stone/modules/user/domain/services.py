from typing import Optional, List, Tuple

from stone.app.request_context import request_context
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

        current_user = request_context.current_user
        kwargs['creator'] = current_user.username
        new_user = UserEntity.parse_obj(kwargs)
        success, message = self.repository.save(new_user)
        if success:
            user = self.repository.get_user_by_name(kwargs['username'])
            return user, message
        return None, message

    def get_users(self, **kwargs) -> Tuple[List[UserEntity], dict, str]:
        page = kwargs['page']
        per_page = kwargs['per_page']
        names_to_exclude = ['sa']
        users, pagination = self.repository.get_users(page, per_page, names_to_exclude)
        return users, pagination, 'Get users succeed'

    def get_user(self, userid: int) -> Tuple[Optional[UserEntity], str]:
        user = self.repository.get_user_by_pk(userid)
        if not user:
            return None, 'no user'
        return user, 'get user success'

    def update_user(self, **kwargs) -> Tuple[Optional[UserEntity], str]:
        user_id = kwargs['user_id']
        user_data = kwargs['user_data']
        user_data['password_hash'] = hash_password(user_data['password'])
        del user_data['password']

        updated_count, message = self.repository.update_user(user_id, user_data)
        if updated_count:
            new_dataset = self.repository.get_user_by_pk(user_id)
            return new_dataset, 'Update user succeed'
        return None, 'Update user failed'

    def delete_user(self, userid: int) -> Tuple[int, str]:
        deleted_count = self.repository.delete_user_by_pk(userid)
        if deleted_count:
            return deleted_count, 'Delete user succeed'
        return deleted_count, 'No user deleted'

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
