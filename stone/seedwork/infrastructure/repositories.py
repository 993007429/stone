from abc import abstractmethod
from contextvars import ContextVar
from typing import Generic, Optional, Type, Union, List

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.seedwork.domain.entities import E
from stone.seedwork.domain.repositories import SingleModelRepository
from stone.seedwork.infrastructure.models import Base


class SQLAlchemyRepository(object):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s


class SQLAlchemySingleModelRepository(SingleModelRepository, SQLAlchemyRepository, Generic[E]):

    @property
    @abstractmethod
    def model_class(self) -> Type[Base]:
        ...

    def save(self, entity: E) -> Optional[E]:
        model = self.model_class(**entity.dict())
        try:
            self.session.add(model)
            self.session.flush([model])
        except IntegrityError:
            return None
        return E.from_orm(model)

    def batch_save(self, entities: List[E]) -> bool:
        for entity in entities:
            self.save(entity)
        return True

    def get(self, pk: int) -> Optional[E]:
        query = self.session.query(self.model_class).filter(self.model_class.id == pk)
        model = query.first()
        if not model:
            return None
        return E.from_orm(model)

    def gets(self, ids: Optional[Union[list, set]]) -> List[Optional[E]]:
        query = self.session.query(self.model_class)
        if ids:
            query = query.filter(self.model_class.in_(ids))
        models = query.all()
        return [E.from_orm(model) for model in models]

    def update(self, pk: int, update_data: dict) -> int:
        updated_count = self.session.query(self.model_class).filter(self.model_class.id == pk).update(update_data, synchronize_session=False)
        return updated_count

    def batch_update(self, ids: Union[list, set], update_data: dict) -> int:
        updated_count = self.session.query(self.model_class).filter(self.model_class.in_(ids)).update(update_data, synchronize_session=False)
        return updated_count

    def delete(self, pk: int) -> int:
        deleted_count = self.session.query(self.model_class).filter(self.model_class.id == pk).delete(synchronize_session=False)
        return deleted_count

    def batch_delete(self, ids: Union[list, set]) -> int:
        deleted_count = self.session.query(self.model_class).filter(self.model_class.id.in_(ids)).delete(synchronize_session=False)
        return deleted_count
