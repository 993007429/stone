import math
from abc import abstractmethod
from contextvars import ContextVar
from typing import Generic, Optional, Type, Union, List, Tuple

from sqlalchemy import and_, not_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.seedwork.domain.entities import E, BaseEntity
from stone.seedwork.domain.enum import LogicType, Condition
from stone.seedwork.domain.repositories import SingleModelRepository
from stone.seedwork.infrastructure.models import Base


class SQLAlchemyRepository(object):

    def __init__(self, session: ContextVar):
        self.session_cv = session

    @property
    def session(self) -> Session:
        s = self.session_cv.get()
        assert s is not None
        return s


class SQLAlchemySingleModelRepository(SingleModelRepository, SQLAlchemyRepository, Generic[E]):

    @property
    @abstractmethod
    def model_class(self) -> Type[Base]:
        ...

    @property
    @abstractmethod
    def entity_class(self) -> Type[BaseEntity]:
        ...

    def save(self, entity: E) -> Optional[E]:
        model = self.model_class(**entity.dict())
        try:
            self.session.add(model)
            self.session.flush([model])
        except IntegrityError:
            return None
        return self.entity_class.from_orm(model)

    def batch_save(self, entities: List[E]) -> bool:
        self.session.bulk_insert_mappings(self.model_class, [entity.dict() for entity in entities])
        return True

    def get(self, pk: int) -> Optional[E]:
        query = self.session.query(self.model_class).filter(self.model_class.id == pk)
        model = query.first()
        if not model:
            return None
        return self.entity_class.from_orm(model)

    def gets(self, ids: Optional[Union[list, set]]) -> List[Optional[E]]:
        query = self.session.query(self.model_class)
        if ids:
            query = query.filter(self.model_class.in_(ids))
        models = query.all()
        return [self.entity_class.from_orm(model) for model in models]

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

    def filter(self, page: int, per_page: int, filters: list, logic: str = LogicType.and_.value, ids: Union[list, set] = None) -> Tuple[List[E], dict]:
        query = self.session.query(self.model_class)
        total = query.count()
        if ids:
            query = query.filter(self.model_class.id.in_(ids))

        for filter_ in filters:
            field = filter_['field']
            condition = filter_['condition']
            value = filter_['value']
            if field in ['create_at']:
                if logic == LogicType.and_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(and_(getattr(self.model_class, field) == value))
                    elif condition == Condition.greater_than.value:
                        query = query.filter(and_(getattr(self.model_class, field).__gt__(value)))
                    elif condition == Condition.less_than.value:
                        query = query.filter(and_(getattr(self.model_class, field).__lt__(value)))
                    elif condition == Condition.is_null.value:
                        query = query.filter(and_(not_(getattr(self.model_class, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(and_(not_(getattr(self.model_class, field).is_not(None))))
                elif logic == LogicType.or_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(or_(getattr(self.model_class, field) == value))
                    elif condition == Condition.greater_than.value:
                        query = query.filter(and_(getattr(self.model_class, field) != value))
                    elif condition == Condition.less_than.value:
                        query = query.filter(and_(getattr(self.model_class, field).contains(value)))
                    elif condition == Condition.is_null.value:
                        query = query.filter(or_(not_(getattr(self.model_class, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(or_(not_(getattr(self.model_class, field).is_not(None))))
            else:
                if logic == LogicType.and_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(and_(getattr(self.model_class, field) == value))
                    elif condition == Condition.unequal.value:
                        query = query.filter(and_(getattr(self.model_class, field) != value))
                    elif condition == Condition.contain.value:
                        query = query.filter(and_(getattr(self.model_class, field).contains(value)))
                    elif condition == Condition.not_contain.value:
                        query = query.filter(and_(not_(getattr(self.model_class, field).contains(value))))
                    elif condition == Condition.is_null.value:
                        query = query.filter(and_(not_(getattr(self.model_class, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(and_(not_(getattr(self.model_class, field).is_not(None))))
                elif logic == LogicType.or_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(or_(getattr(self.model_class, field) == value))
                    elif condition == Condition.unequal.value:
                        query = query.filter(or_(getattr(self.model_class, field) != value))
                    elif condition == Condition.contain.value:
                        query = query.filter(or_(getattr(self.model_class, field).contains(value)))
                    elif condition == Condition.not_contain.value:
                        query = query.filter(or_(not_(getattr(self.model_class, field).contains(value))))
                    elif condition == Condition.is_null.value:
                        query = query.filter(or_(not_(getattr(self.model_class, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(or_(not_(getattr(self.model_class, field).is_not(None))))

        query = query.order_by(self.model_class.id)

        offset = min((page - 1), math.floor(total / per_page)) * per_page
        query = query.offset(offset).limit(per_page)

        entities = [self.entity_class.from_orm(model) for model in query.all()]
        pagination = {'total': total, 'page': page, 'per_page': per_page}

        return entities, pagination
