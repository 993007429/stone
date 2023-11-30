import math
from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy import not_, and_, or_, desc, exists
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.modules.slice.domain.entities import SliceEntity, LabelEntity, SliceLabelEntity, DataSetEntity, \
    DataSetSliceEntity
from stone.modules.slice.domain.repositories import SliceRepository
from stone.modules.slice.domain.value_objects import Condition, LogicType
from stone.modules.slice.infrastructure.models import Slice, Label, SliceLabel, DataSet, DataSetSlice


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

    def get_dataset_by_id(self, pk: int) -> Optional[DataSetEntity]:
        query = self._session.query(DataSet).filter(DataSet.id == pk, DataSet.is_deleted.is_(False))
        model = query.first()
        if not model:
            return None
        return DataSetEntity.from_orm(model)

    def get_label_by_name(self, name: str) -> Optional[LabelEntity]:
        query = self._session.query(Label).filter_by(name=name)
        model = query.first()
        if not model:
            return None
        return LabelEntity(**model.dict)

    def get_slice_fields(self) -> list:
        columns_with_types = {column_name: str(column.type) for column_name, column in Slice.__table__.columns.items()}
        slice_fields = []
        for column_name, column_type in columns_with_types.items():
            slice_fields.append(column_name)
        return slice_fields

    def get_slice_labels_by_slice(self, slice_id: int) -> List[SliceLabelEntity]:
        query = self._session.query(SliceLabel).filter(
            SliceLabel.slice_id == slice_id, SliceLabel.is_deleted.is_(False))
        models = query.order_by(SliceLabel.label_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_slice_labels_by_label(self, label_id: int) -> List[SliceLabelEntity]:
        query = self._session.query(SliceLabel).filter(
            SliceLabel.label_id == label_id, SliceLabel.is_deleted.is_(False))
        models = query.order_by(SliceLabel.slice_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_slice_labels_by_label_ids(self, label_ids: list) -> List[SliceLabelEntity]:
        query = self._session.query(SliceLabel).filter(
            SliceLabel.label_id.in_(label_ids), SliceLabel.is_deleted.is_(False))
        models = query.order_by(SliceLabel.slice_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_dataset_slices_by_dataset(self, dataset_id: int) -> List[DataSetSliceEntity]:
        query = self._session.query(DataSetSlice).filter(
            DataSetSlice.dataset_id == dataset_id, DataSetSlice.is_deleted.is_(False))
        models = query.order_by(DataSetSlice.dataset_id).all()
        return [DataSetSliceEntity.from_orm(model) for model in models]

    def delete_label(self, label_id: int) -> int:
        deleted_count = self._session.query(Label).filter(Label.id == label_id).update(
            {'is_deleted': 1}, synchronize_session=False)

        self._session.query(SliceLabel).filter(SliceLabel.label_id == label_id).update(
            {'is_deleted': 1}, synchronize_session=False)
        return deleted_count

    def delete_dataset(self, dataset_id: int) -> int:
        deleted_count = self._session.query(DataSet).filter(DataSet.id == dataset_id).update(
            {'is_deleted': 1}, synchronize_session=False)

        self._session.query(DataSetSlice).filter(DataSetSlice.dataset_id == dataset_id).update(
            {'is_deleted': 1}, synchronize_session=False)
        return deleted_count

    def copy_dataset(self, dataset_id: int) -> Optional[DataSetEntity]:
        deleted_count = self._session.query(DataSet).filter(DataSet.id == dataset_id).update(
            {'is_deleted': 1}, synchronize_session=False)

        self._session.query(DataSetSlice).filter(DataSetSlice.dataset_id == dataset_id).update(
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

    def update_dataset(self, dataset_id: int, dataset_data: dict) -> Tuple[int, str]:
        updated_count = self._session.query(DataSet).filter(DataSet.id == dataset_id).update(
            dataset_data, synchronize_session=False)
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

    def add_slices(self, dataset_id: list, slice_ids: list) -> bool:
        dataset = self._session.query(DataSet).filter(DataSet.id == dataset_id).first()
        slices = self._session.query(Slice).filter(Slice.id.in_(slice_ids)).all()

        dataset_slices = []
        for slice_ in slices:
            dataset_slice = DataSetSlice(dataset_id=dataset.id, slice_id=slice_.id)
            dataset_slices.append(dataset_slice)
        self._session.add_all(dataset_slices)
        return True

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

    def save_data_set(self, entity: DataSetEntity) -> Tuple[bool, Optional[DataSetEntity]]:
        model = DataSet(**entity.dict())
        try:
            self._session.add(model)
            self._session.flush([model])
        except IntegrityError as e:
            # self._session.rollback()
            return False, None
        return True, entity.from_orm(model)

    def filter_slices(self, page: int, per_page: int, logic: str, filters: list, slice_ids: set) -> Tuple[List[SliceEntity], dict]:
        query = self._session.query(Slice).filter(Slice.is_deleted.is_(False))
        total = query.count()
        if slice_ids:
            query = query.filter(Slice.id.in_(slice_ids))

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
        query = self._session.query(Label).filter(Label.is_deleted.is_(False))
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

    def filter_datasets(self, page: int, per_page: int, filters: list) -> Tuple[List[DataSetEntity], dict]:
        query = self._session.query(DataSet).filter(DataSet.is_deleted.is_(False))
        for filter_ in filters:
            field = filter_['field']
            condition = filter_['condition']
            value = filter_['value']
            if field in ['create_at']:
                if condition == 'equal':
                    query = query.filter(and_(getattr(DataSet, field) == value))
                elif condition == 'greater_than':
                    query = query.filter(and_(getattr(DataSet, field).__gt__(value)))
                elif condition == 'less_than':
                    query = query.filter(and_(getattr(DataSet, field).__lt__(value)))
                elif condition == 'is_null':
                    query = query.filter(and_(not_(getattr(DataSet, field).is_(None))))
                elif condition == 'not_null':
                    query = query.filter(and_(not_(getattr(DataSet, field).is_not(None))))
            else:
                if condition == 'equal':
                    query = query.filter(and_(getattr(DataSet, field) == value))
                elif condition == 'unequal':
                    query = query.filter(and_(getattr(DataSet, field) != value))
                elif condition == 'contain':
                    query = query.filter(and_(getattr(DataSet, field).contains(value)))
                elif condition == 'not_contain':
                    query = query.filter(and_(not_(getattr(DataSet, field).contains(value))))
                elif condition == 'is_null':
                    query = query.filter(and_(not_(getattr(DataSet, field).is_(None))))
                elif condition == 'not_null':
                    query = query.filter(and_(not_(getattr(DataSet, field).is_not(None))))

        query = query.order_by(DataSet.id)
        total = query.count()

        offset = min((page - 1), math.floor(total / per_page)) * per_page
        query = query.offset(offset).limit(per_page)

        pagination = {
            'total': total,
            'page': page,
            'per_page': per_page
        }

        return [DataSetEntity.from_orm(model) for model in query.all()], pagination

    def get_datasets_with_fuzzy(self, userid: int, name: Optional[str]) -> List[DataSetEntity]:
        query = self._session.query(DataSet).filter(DataSet.userid == userid, DataSet.is_deleted.is_(False))
        if name:
            query = query.filter(DataSet.name.like(f'{name}%'))
        return [DataSetEntity.from_orm(model) for model in query.all()]

    def get_labels_with_fuzzy(self, name: Optional[str]) -> List[LabelEntity]:
        query = self._session.query(Label).filter(Label.is_deleted.is_(False))
        if name:
            query = query.filter(Label.name.like(f'{name}%'))
        return [LabelEntity.from_orm(model) for model in query.all()]











