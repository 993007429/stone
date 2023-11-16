from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.user.domain.entities import UserEntity
from src.modules.user.domain.repositories import UserRepository
from src.modules.user.infrastructure.models import User


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
        self._session.begin()
        try:
            self._session.add(model)
            self._session.flush([model])
            self._session.commit()
        except IntegrityError as e:
            return False, 'Duplicate username '
        return True, 'Create user success'

    def update(self, pk: int, entity: UserEntity) -> Tuple[Optional[UserEntity], str]:
        query = self._session.query(User).filter_by(id=pk)
        model = query.first()
        if not model:
            return None, 'no user'

        model = User(**entity.dict())
        self._session.begin()
        try:
            self._session.add(model)
            self._session.flush([model])
            self._session.commit()
        except IntegrityError as e:
            return False, 'Duplicate username '
        return True, 'Create user success'

    def gets(self) -> List[Optional[UserEntity]]:
        self._session.begin()
        query = self._session.query(User)
        models = query.all()
        self._session.commit()
        return [UserEntity(**model.dict) for model in models]

    def get_user_by_name(self, username: str) -> Optional[UserEntity]:
        self._session.begin()
        query = self._session.query(User).filter_by(username=username)
        model = query.first()
        self._session.commit()
        if not model:
            return None
        return UserEntity(**model.dict)

    def get_user_by_pk(self, pk: int) -> Optional[UserEntity]:
        self._session.begin()
        query = self._session.query(User).filter_by(id=pk)
        model = query.first()
        self._session.commit()
        if not model:
            return None
        return UserEntity(**model.dict)


