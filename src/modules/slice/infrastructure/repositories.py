import math
from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy import not_, and_, or_, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.slice.domain.entities import SliceEntity
from src.modules.slice.domain.repositories import SliceRepository
from src.modules.slice.infrastructure.models import Slice


class SQLAlchemySliceRepository(SliceRepository):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    def get_slice_by_id(self, pk: int) -> Optional[SliceEntity]:
        self._session.begin()
        query = self._session.query(Slice).filter_by(id=pk)
        model = query.first()
        self._session.commit()
        if not model:
            return None
        return SliceEntity(**model.dict)

    def filter_slices(self, **kwargs) -> Tuple[List[SliceEntity], dict]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']

        self._session.begin()
        query = self._session.query(Slice)

        for filter_ in filters:
            field = filter_['field']
            condition = filter_['condition']
            value = filter_['value']
            if field in ['create_at']:
                if logic == 'and':
                    if condition == 'equal':
                        query = query.filter(and_(getattr(Slice, field) == value))
                    elif condition == 'greater_than':
                        query = query.filter(and_(getattr(Slice, field).__gt__(value)))
                    elif condition == 'less_than':
                        query = query.filter(and_(getattr(Slice, field).__lt__(value)))
                    elif condition == 'is_null':
                        query = query.filter(and_(not_(getattr(Slice, field).is_(None))))
                    elif condition == 'not_null':
                        query = query.filter(and_(not_(getattr(Slice, field).is_not(None))))
                elif logic == 'or':
                    if condition == 'equal':
                        query = query.filter(or_(getattr(Slice, field) == value))
                    elif condition == 'greater_than':
                        query = query.filter(and_(getattr(Slice, field) != value))
                    elif condition == 'less_than':
                        query = query.filter(and_(getattr(Slice, field).contains(value)))
                    elif condition == 'is_null':
                        query = query.filter(or_(not_(getattr(Slice, field).is_(None))))
                    elif condition == 'not_null':
                        query = query.filter(or_(not_(getattr(Slice, field).is_not(None))))
            else:
                if logic == 'and':
                    if condition == 'equal':
                        query = query.filter(and_(getattr(Slice, field) == value))
                    elif condition == 'unequal':
                        query = query.filter(and_(getattr(Slice, field) != value))
                    elif condition == 'contain':
                        query = query.filter(and_(getattr(Slice, field).contains(value)))
                    elif condition == 'not_contain':
                        query = query.filter(and_(not_(getattr(Slice, field).contains(value))))
                    elif condition == 'is_null':
                        query = query.filter(and_(not_(getattr(Slice, field).is_(None))))
                    elif condition == 'not_null':
                        query = query.filter(and_(not_(getattr(Slice, field).is_not(None))))
                elif logic == 'or':
                    if condition == 'equal':
                        query = query.filter(or_(getattr(Slice, field) == value))
                    elif condition == 'unequal':
                        query = query.filter(or_(getattr(Slice, field) != value))
                    elif condition == 'contain':
                        query = query.filter(or_(getattr(Slice, field).contains(value)))
                    elif condition == 'not_contain':
                        query = query.filter(or_(not_(getattr(Slice, field).contains(value))))
                    elif condition == 'is_null':
                        query = query.filter(or_(not_(getattr(Slice, field).is_(None))))
                    elif condition == 'not_null':
                        query = query.filter(or_(not_(getattr(Slice, field).is_not(None))))

        query = query.order_by(desc(Slice.id))
        total = query.count()

        page = min(page, math.ceil(total / per_page))
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page)

        pagination = {
            'total': total,
            'page': page,
            'per_page': per_page
        }

        slices = []
        for model in query.all():
            entity = SliceEntity.from_orm(model)
            slices.append(entity)

        return slices, pagination

    def save(self, entity: SliceEntity) -> Tuple[bool, SliceEntity]:
        model = Slice(**entity.dict())
        self._session.begin()
        self._session.add(model)
        self._session.flush([model])
        self._session.commit()
        return True, entity.from_orm(model)















