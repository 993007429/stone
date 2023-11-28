import math
from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy import not_, and_, or_, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.slice.domain.entities import SliceEntity, LabelEntity
from src.modules.slice.domain.repositories import SliceRepository
from src.modules.slice.infrastructure.models import Slice, Label, SliceLabel


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

    def filter_labels(self, **kwargs) -> Tuple[List[LabelEntity], dict]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']

        self._session.begin()
        query = self._session.query(Label)

        for filter_ in filters:
            field = filter_['field']
            condition = filter_['condition']
            value = filter_['value']
            if field in ['create_at']:
                if condition == 'equal':
                    query = query.filter(and_(getattr(Label, field) == value))
                elif condition == 'greater_than':
                    query = query.filter(and_(getattr(Label, field).__gt__(value)))
                elif condition == 'less_than':
                    query = query.filter(and_(getattr(Label, field).__lt__(value)))
                elif condition == 'is_null':
                    query = query.filter(and_(not_(getattr(Label, field).is_(None))))
                elif condition == 'not_null':
                    query = query.filter(and_(not_(getattr(Label, field).is_not(None))))
            else:
                if condition == 'equal':
                    query = query.filter(and_(getattr(Label, field) == value))
                elif condition == 'unequal':
                    query = query.filter(and_(getattr(Label, field) != value))
                elif condition == 'contain':
                    query = query.filter(and_(getattr(Label, field).contains(value)))
                elif condition == 'not_contain':
                    query = query.filter(and_(not_(getattr(Label, field).contains(value))))
                elif condition == 'is_null':
                    query = query.filter(and_(not_(getattr(Label, field).is_(None))))
                elif condition == 'not_null':
                    query = query.filter(and_(not_(getattr(Label, field).is_not(None))))

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

        labels = []
        for model in query.all():
            entity = LabelEntity.from_orm(model)
            labels.append(entity)

        return labels, pagination

    def delete_slices(self, **kwargs) -> int:
        ids = kwargs['ids']
        self._session.begin()
        deleted_count = self._session.query(Slice).filter(Slice.id.in_(ids)).update(
            {'is_deleted': 1}, synchronize_session=False)
        self._session.commit()
        return deleted_count

    def delete_labels(self, **kwargs) -> int:
        ids = kwargs['ids']
        self._session.begin()
        deleted_count = self._session.query(Label).filter(Label.id.in_(ids)).update(
            {'is_deleted': 1}, synchronize_session=False)
        self._session.commit()
        return deleted_count

    def update_slices(self, **kwargs) -> int:
        ids = kwargs['ids']
        del kwargs['ids']
        self._session.begin()
        updated_count = self._session.query(Slice).filter(Slice.id.in_(ids)).update(
            kwargs, synchronize_session=False)
        self._session.commit()
        return updated_count

    def update_labels(self, **kwargs) -> int:
        ids = kwargs['ids']
        del kwargs['ids']
        self._session.begin()
        updated_count = self._session.query(Label).filter(Label.id.in_(ids)).update(
            kwargs, synchronize_session=False)
        self._session.commit()
        return updated_count

    def add_labels(self, **kwargs) -> int:
        ids = kwargs['ids']
        label_ids = kwargs['label_ids']

        self._session.begin()
        slices = self._session.query(Slice).filter(Slice.id.in_(ids)).all()
        labels = self._session.query(Label).filter(Label.id.in_(label_ids)).all()

        slice_labels = []
        for slice_ in slices:
            for label in labels:
                slice_label = SliceLabel(slice_id=slice_.id, label_id=label.id, label_name=label.name)
                slice_labels.append(slice_label)

        self._session.add_all(slice_labels)
        self._session.commit()
        return len(ids)

    def save_slice(self, entity: SliceEntity) -> Tuple[bool, Optional[SliceEntity, str]]:
        model = Slice(**entity.dict())
        self._session.begin()
        try:
            self._session.add(model)
            self._session.flush([model])
            self._session.commit()
        except IntegrityError as e:
            return False, 'Duplicate slice '
        return True, entity.from_orm(model)

    def save_label(self, entity: LabelEntity) -> Tuple[bool, Optional[LabelEntity, str]]:
        model = Label(**entity.dict())
        self._session.begin()
        try:
            self._session.add(model)
            self._session.flush([model])
            self._session.commit()
        except IntegrityError as e:
            return False, 'Duplicate label'
        return True, entity.from_orm(model)
















