from abc import abstractmethod, ABCMeta
from contextvars import ContextVar
from typing import Generic, Optional, Type, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.seedwork.domain.entities import E
from stone.seedwork.infrastructure.models import Base


class SQLAlchemyRepository(object):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s


class SQLAlchemySingleModelRepository(SQLAlchemyRepository, Generic[E]):

    @property
    @abstractmethod
    def model_class(self) -> Type[Base]:
        ...

    def save_entity(self, entity: E) -> bool:
        model = self.model_class(**entity.dict())
        try:
            self.session.add(model)
            self.session.flush([model])
        except IntegrityError:
            return False
        return True

    def get_entity_by_pk(self, pk: int) -> Optional[E]:
        query = self.session.query(self.model_class).filter(self.model_class.id == pk)
        model = query.first()
        if not model:
            return None
        return E.from_orm(model)

    def update_entity_by_pk(self, pk: int, update_data: dict) -> int:
        updated_count = self.session.query(self.model_class).filter(self.model_class.id == pk).update(update_data, synchronize_session=False)
        return updated_count

    def delete_entity_by_pk(self, pk: int) -> int:
        deleted_count = self.session.query(self.model_class).filter(self.model_class.id == pk).delete(synchronize_session=False)
        return deleted_count
