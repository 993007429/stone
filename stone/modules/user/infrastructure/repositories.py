import math
from typing import List, Optional, Tuple, Union, Type

from sqlalchemy.exc import IntegrityError

from stone.modules.user.domain.entities import UserEntity
from stone.modules.user.domain.repositories import UserRepository
from stone.modules.user.infrastructure.models import User
from stone.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SQLAlchemyUserRepository(SQLAlchemySingleModelRepository[UserEntity]):

    @property
    def model_class(self) -> Type[User]:
        return User

    def get_user_by_name(self, username: str) -> Optional[UserEntity]:
        query = self.session.query(User).filter(User.username == username)
        model = query.first()
        if not model:
            return None
        return UserEntity.from_orm(model)

    def get_users(self, page: int, per_page: int, names_to_exclude: Union[list, set]) -> Tuple[List[UserEntity], dict]:
        query = self.session.query(User).filter(User.username.not_in(names_to_exclude)).order_by(User.id)
        total = query.count()

        offset = min((page - 1), math.floor(total / per_page)) * per_page
        query = query.offset(offset).limit(per_page)

        return [UserEntity.from_orm(model) for model in query.all()], {'total': total, 'page': page, 'per_page': per_page}
