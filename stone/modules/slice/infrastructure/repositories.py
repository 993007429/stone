import math
from typing import List, Optional, Tuple, Union, Type

from sqlalchemy import not_, and_, or_
from sqlalchemy.exc import IntegrityError

from stone.modules.slice.domain.entities import SliceEntity, LabelEntity, SliceLabelEntity, DataSetEntity, \
    DataSetSliceEntity, FilterTemplateEntity
from stone.modules.slice.domain.enum import Condition, LogicType
from stone.modules.slice.domain.repositories import SliceRepository, DataSetRepository, LabelRepository, \
    FilterTemplateRepository
from stone.modules.slice.infrastructure.models import Slice, Label, SliceLabel, DataSet, DataSetSlice, FilterTemplate
from stone.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SQLAlchemySliceRepository(SliceRepository, SQLAlchemySingleModelRepository[SliceEntity]):

    @property
    def model_class(self) -> Type[Slice]:
        return Slice

    def delete_slices(self, ids: list) -> int:
        deleted_count = self.session.query(Slice).filter(Slice.id.in_(ids)).delete(synchronize_session=False)
        return deleted_count

    def get_slice_fields(self) -> list:
        columns_with_types = {column_name: str(column.type) for column_name, column in Slice.__table__.columns.items()}
        slice_fields = []
        for column_name, column_type in columns_with_types.items():
            slice_fields.append(column_name)
        return slice_fields

    def get_slice_labels_by_slice(self, slice_id: int) -> List[SliceLabelEntity]:
        query = self.session.query(SliceLabel).filter(SliceLabel.slice_id == slice_id)
        models = query.order_by(SliceLabel.label_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_slice_labels_by_label(self, label_id: int) -> List[SliceLabelEntity]:
        query = self.session.query(SliceLabel).filter(SliceLabel.label_id == label_id)
        models = query.order_by(SliceLabel.slice_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_slice_labels_by_slice_ids(self, slice_ids: Union[list, set]) -> List[SliceLabelEntity]:
        query = self.session.query(SliceLabel).filter(SliceLabel.slice_id.in_(slice_ids))
        models = query.order_by(SliceLabel.label_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def get_slice_labels_by_label_ids(self, label_ids: list) -> List[SliceLabelEntity]:
        query = self.session.query(SliceLabel).filter(SliceLabel.label_id.in_(label_ids))
        models = query.order_by(SliceLabel.slice_id).all()
        return [SliceLabelEntity.from_orm(model) for model in models]

    def update_slices(self, ids: list, update_data: dict) -> int:
        updated_count = self.session.query(Slice).filter(Slice.id.in_(ids)).update(update_data, synchronize_session=False)
        return updated_count

    def add_labels(self, slice_ids: list, label_ids: list) -> int:
        slices = self.session.query(Slice).filter(Slice.id.in_(slice_ids)).all()
        labels = self.session.query(Label).filter(Label.id.in_(label_ids)).all()

        slice_labels = []
        for slice_ in slices:
            for label in labels:
                slice_label = SliceLabel(slice_id=slice_.id, label_id=label.id, label_name=label.name)
                slice_labels.append(slice_label)
        self.session.add_all(slice_labels)
        return len(slice_ids)


class SQLAlchemyDataSetRepository(DataSetRepository, SQLAlchemySingleModelRepository[DataSetEntity]):

    @property
    def model_class(self) -> Type[DataSet]:
        return DataSet

    def get_datasets_with_fuzzy(self, userid: int, name: Optional[str]) -> List[DataSetEntity]:
        query = self.session.query(DataSet).filter(DataSet.userid == userid)
        if name:
            query = query.filter(DataSet.name.like(f'{name}%'))
        return [DataSetEntity.from_orm(model) for model in query.all()]

    def get_dataset_slices_by_dataset(self, dataset_id: int) -> List[DataSetSliceEntity]:
        query = self.session.query(DataSetSlice).filter(DataSetSlice.dataset_id == dataset_id)
        models = query.order_by(DataSetSlice.dataset_id).all()
        return [DataSetSliceEntity.from_orm(model) for model in models]

    def add_slices(self, dataset_id: list, slice_ids: list) -> int:
        dataset = self.session.query(DataSet).filter(DataSet.id == dataset_id).first()
        slices = self.session.query(Slice).filter(Slice.id.in_(slice_ids)).all()

        dataset_slices = []
        for slice_ in slices:
            dataset_slice = DataSetSlice(dataset_id=dataset.id, slice_id=slice_.id)
            dataset_slices.append(dataset_slice)
        self.session.add_all(dataset_slices)
        return len(slices)

    def batch_save_dataset_slice(self, entities: List[DataSetSliceEntity]) -> bool:
        models = [DataSetSlice(**entity.dict()) for entity in entities]
        try:
            self.session.add_all(models)
            self.session.flush(models)
        except IntegrityError:
            return False
        return True

    def delete_dataset(self, dataset_id: int) -> int:
        deleted_count = self.session.query(DataSet).filter(DataSet.id == dataset_id).delete(synchronize_session=False)
        self.session.query(DataSetSlice).filter(DataSetSlice.dataset_id == dataset_id).delete(synchronize_session=False)
        return deleted_count

    def delete_dataset_slices(self, dataset_id: int, slice_ids: list) -> int:
        deleted_count = self.session.query(DataSetSlice).filter(
            DataSetSlice.dataset_id == dataset_id,
            DataSetSlice.slice_id.in_(slice_ids)).delete(synchronize_session=False)
        return deleted_count


class SQLAlchemyLabelRepository(LabelRepository, SQLAlchemySingleModelRepository[LabelEntity]):

    @property
    def model_class(self) -> Type[Label]:
        return Label

    def get_label_by_name(self, name: str) -> Optional[LabelEntity]:
        query = self.session.query(Label).filter_by(name=name)
        model = query.first()
        if not model:
            return None
        return LabelEntity.from_orm(model)

    def update_label(self, label_id: int, label_data: dict) -> Tuple[int, str]:
        to_update_model_name = self.session.query(Label).filter(Label.id == label_id).first().name

        name = label_data.get('name')
        if name and name != to_update_model_name:
            self.session.query(SliceLabel).filter(SliceLabel.label_id == label_id).update({'label_name': name}, synchronize_session=False)

        updated_count = self.session.query(Label).filter(Label.id == label_id).update({'name': name}, synchronize_session=False)
        return updated_count, 'Update label succeed'

    def get_labels_with_fuzzy(self, name: Optional[str]) -> List[LabelEntity]:
        query = self.session.query(Label)
        if name:
            query = query.filter(Label.name.like(f'{name}%'))
        return [LabelEntity.from_orm(model) for model in query.all()]

    def delete_label(self, label_id: int) -> int:
        deleted_count = self.session.query(Label).filter(Label.id == label_id).delete(synchronize_session=False)
        self.session.query(SliceLabel).filter(SliceLabel.label_id == label_id).delete(synchronize_session=False)
        return deleted_count


class SQLAlchemyFilterTemplateRepository(FilterTemplateRepository, SQLAlchemySingleModelRepository[FilterTemplateEntity]):

    @property
    def model_class(self) -> Type[FilterTemplate]:
        return FilterTemplate
