from contextvars import ContextVar
from typing import List, Optional

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

    def save(self, entity: UserEntity) -> bool:
        model = User(**entity.dict)

        self._session.begin()
        self._session.add(model)
        self._session.flush([model])
        self._session.commit()

        return True

    def gets(self) -> List[Optional[UserEntity]]:
        query = self._session.query(User)
        models = query.all()
        return [UserEntity(**model.dict) for model in models]



