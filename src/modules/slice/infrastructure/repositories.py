import math
from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy import not_, and_, or_, desc, exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.slice.domain.entities import SliceEntity, LabelEntity, SliceLabelEntity
from src.modules.slice.domain.repositories import SliceRepository
from src.modules.slice.domain.value_objects import Condition, LogicType
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
        query = self._session.query(Slice).filter(Slice.id == pk, Slice.is_deleted.is_(False))
        model = query.first()
        if not model:
            return None
        return SliceEntity(**model.dict)

    def get_slices(self, ids: list) -> List[SliceEntity]:
        query = self._session.query(Slice).filter(Slice.id.in_(ids), Slice.is_deleted.is_(False))
        models = query.all()
        return [SliceEntity.from_orm(model) for model in models]

    def delete_slices(self, ids: list) -> int:
        deleted_count = self._session.query(Slice).filter(Slice.id.in_(ids)).update(
            {'is_deleted': 1}, synchronize_session=False)
        return deleted_count

    def get_label_by_id(self, pk: int) -> Optional[LabelEntity]:
        query = self._session.query(Label).filter(Label.id == pk, Label.is_deleted.is_(False))
        model = query.first()
        if not model:
            return None
        return LabelEntity(**model.dict)

    def get_label_by_name(self, name: str) -> Optional[LabelEntity]:
        query = self._session.query(Label).filter_by(name=name)
        model = query.first()
        if not model:
            return None
        return LabelEntity(**model.dict)

    def get_slice_labels_by_slice(self, slice_id: int) -> List[SliceLabelEntity]:
        query = self._session.query(SliceLabel).filter(
            SliceLabel.slice_id == slice_id, SliceLabel.is_deleted.is_(False))
        models = query.order_by(SliceLabel.label_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def delete_label(self, label_id: int) -> int:
        deleted_count = self._session.query(Label).filter(Label.id == label_id).update(
            {'is_deleted': 1}, synchronize_session=False)

        self._session.query(SliceLabel).filter(SliceLabel.label_id == label_id).update(
            {'is_deleted': 1}, synchronize_session=False)
        return deleted_count

    def update_slices(self, ids: list, update_data: dict) -> int:
        updated_count = self._session.query(Slice).filter(Slice.id.in_(ids)).update(
            update_data, synchronize_session=False)
        return updated_count

    def update_label(self, label_id: int, label_data: dict) -> Tuple[int, str]:
        to_update_model_name = self._session.query(Label).filter(Label.id == label_id).first().name

        name = label_data.get('name')
        if name and name != to_update_model_name:
            self._session.query(SliceLabel).filter(SliceLabel.label_id == label_id).update(
                {'label_name': name}, synchronize_session=False)

        updated_count = self._session.query(Label).filter(Label.id == label_id).update(
            {'name': name}, synchronize_session=False)
        return updated_count, 'Update label succeed'

    def add_labels(self, slice_ids: list, label_ids: list) -> int:
        slices = self._session.query(Slice).filter(Slice.id.in_(slice_ids)).all()
        labels = self._session.query(Label).filter(Label.id.in_(label_ids)).all()

        slice_labels = []
        for slice_ in slices:
            for label in labels:
                slice_label = SliceLabel(slice_id=slice_.id, label_id=label.id, label_name=label.name)
                slice_labels.append(slice_label)
        self._session.add_all(slice_labels)
        return len(slice_ids)

    def save_slice(self, entity: SliceEntity) -> Tuple[bool, Optional[SliceEntity]]:
        model = Slice(**entity.dict())
        try:
            self._session.add(model)
            self._session.flush([model])
        except IntegrityError as e:
            return False, None
        return True, entity.from_orm(model)

    def save_label(self, entity: LabelEntity) -> Tuple[bool, Optional[LabelEntity]]:
        model = Label(**entity.dict())
        try:
            self._session.add(model)
            self._session.flush([model])
        except IntegrityError as e:
            return False, None
        return True, entity.from_orm(model)

    def filter_slices(self, page: int, per_page: int, logic: str, filters: list) -> Tuple[List[SliceEntity], dict]:
        query = self._session.query(Slice)
        for filter_ in filters:
            field = filter_['field']
            condition = filter_['condition']
            value = filter_['value']
            if field in ['create_at']:
                if logic == LogicType.and_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(and_(getattr(Slice, field) == value))
                    elif condition == Condition.greater_than.value:
                        query = query.filter(and_(getattr(Slice, field).__gt__(value)))
                    elif condition == Condition.less_than.value:
                        query = query.filter(and_(getattr(Slice, field).__lt__(value)))
                    elif condition == Condition.is_null.value:
                        query = query.filter(and_(not_(getattr(Slice, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(and_(not_(getattr(Slice, field).is_not(None))))
                elif logic == LogicType.or_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(or_(getattr(Slice, field) == value))
                    elif condition == Condition.greater_than.value:
                        query = query.filter(and_(getattr(Slice, field) != value))
                    elif condition == Condition.less_than.value:
                        query = query.filter(and_(getattr(Slice, field).contains(value)))
                    elif condition == Condition.is_null.value:
                        query = query.filter(or_(not_(getattr(Slice, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(or_(not_(getattr(Slice, field).is_not(None))))
            else:
                if logic == LogicType.and_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(and_(getattr(Slice, field) == value))
                    elif condition == Condition.unequal.value:
                        query = query.filter(and_(getattr(Slice, field) != value))
                    elif condition == Condition.contain.value:
                        query = query.filter(and_(getattr(Slice, field).contains(value)))
                    elif condition == Condition.not_contain.value:
                        query = query.filter(and_(not_(getattr(Slice, field).contains(value))))
                    elif condition == Condition.is_null.value:
                        query = query.filter(and_(not_(getattr(Slice, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(and_(not_(getattr(Slice, field).is_not(None))))
                elif logic == LogicType.or_.value:
                    if condition == Condition.equal.value:
                        query = query.filter(or_(getattr(Slice, field) == value))
                    elif condition == Condition.unequal.value:
                        query = query.filter(or_(getattr(Slice, field) != value))
                    elif condition == Condition.contain.value:
                        query = query.filter(or_(getattr(Slice, field).contains(value)))
                    elif condition == Condition.not_contain.value:
                        query = query.filter(or_(not_(getattr(Slice, field).contains(value))))
                    elif condition == Condition.is_null.value:
                        query = query.filter(or_(not_(getattr(Slice, field).is_(None))))
                    elif condition == Condition.not_null.value:
                        query = query.filter(or_(not_(getattr(Slice, field).is_not(None))))

        query = query.order_by(Slice.id)
        total = query.count()

        offset = min((page - 1), math.floor(total / per_page)) * per_page
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

    def filter_labels(self, page: int, per_page: int, filters: list) -> Tuple[List[LabelEntity], dict]:
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

        query = query.order_by(Label.id)
        total = query.count()

        offset = min((page - 1), math.floor(total / per_page)) * per_page
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
