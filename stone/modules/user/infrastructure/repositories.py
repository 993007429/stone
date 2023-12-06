import math
from contextvars import ContextVar
from typing import List, Optional, Tuple, Union

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.modules.user.domain.entities import UserEntity
from stone.modules.user.domain.repositories import UserRepository
from stone.modules.user.infrastructure.models import User


class SQLAlchemyUserRepository(UserRepository):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    def save(self, entity: UserEntity) -> Tuple[bool, str]:
        model = User(**entity.dict())
        try:
            self._session.add(model)
            self._session.flush([model])
        except IntegrityError:
            return False, 'Duplicate username '
        return True, 'Create user success'

    def update_user(self, user_id: int, user_data: dict) -> Tuple[int, str]:
        updated_count = self._session.query(User).filter(User.id == user_id).update(
            user_data, synchronize_session=False)
        return updated_count, 'Update user succeed'

    def get_users(self, page: int, per_page: int, names_to_exclude: Union[list, set]) -> Tuple[List[UserEntity], dict]:
        query = self._session.query(User).filter(User.username.not_in(names_to_exclude)).order_by(User.id)
        total = query.count()

        offset = min((page - 1), math.floor(total / per_page)) * per_page
        query = query.offset(offset).limit(per_page)

        pagination = {
            'total': total,
            'page': page,
            'per_page': per_page
        }

        users = []
        for model in query.all():
            entity = UserEntity.from_orm(model)
            users.append(entity)

        return users, pagination

    def get_user_by_name(self, username: str) -> Optional[UserEntity]:
        query = self._session.query(User).filter(User.username == username)
        model = query.first()
        if not model:
            return None
        return UserEntity.from_orm(model)

    def get_user_by_pk(self, pk: int) -> Optional[UserEntity]:
        query = self._session.query(User).filter(User.id == pk)
        model = query.first()
        if not model:
            return None
        return UserEntity.from_orm(model)

    def delete_user_by_pk(self, pk: int) -> int:
        deleted_count = self._session.query(User).filter(User.id == pk).delete(synchronize_session=False)
        return deleted_count
