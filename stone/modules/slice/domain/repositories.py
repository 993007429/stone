from abc import ABCMeta

from typing import List, Tuple, Union, Optional

from stone.modules.slice.domain.entities import SliceEntity, SliceLabelEntity, DataSetEntity, DataSetSliceEntity, LabelEntity, FilterTemplateEntity
from stone.seedwork.domain.repositories import SingleModelRepository


class SliceRepository(SingleModelRepository[SliceEntity], metaclass=ABCMeta):

    def delete_slices(self, ids: list) -> int:
        ...

    def get_slice_fields(self) -> list:
        ...

    def get_slice_labels_by_slice(self, slice_id: int) -> List[SliceLabelEntity]:
        ...

    def get_slice_labels_by_label(self, label_id: int) -> List[SliceLabelEntity]:
        ...

    def get_slice_labels_by_slice_ids(self, slice_ids: Union[list, set]) -> List[SliceLabelEntity]:
        ...

    def get_slice_labels_by_label_ids(self, label_ids: list) -> List[SliceLabelEntity]:
        ...

    def update_slices(self, ids: list, update_data: dict) -> int:
        ...

    def add_labels(self, slice_ids: list, label_ids: list) -> int:
        ...

    def filter_slices(self, page: int, per_page: int, logic: str, filters: list, slice_ids: set) -> Tuple[List[SliceEntity], dict]:
        ...


class DataSetRepository(SingleModelRepository[DataSetEntity], metaclass=ABCMeta):

    def get_datasets_with_fuzzy(self, userid: int, name: Optional[str]) -> List[DataSetEntity]:
        ...

    def get_dataset_slices_by_dataset(self, dataset_id: int) -> List[DataSetSliceEntity]:
        ...

    def add_slices(self, dataset_id: list, slice_ids: list) -> int:
        ...

    def filter_datasets(self, page: int, per_page: int, filters: list) -> Tuple[List[DataSetEntity], dict]:
        ...

    def batch_save_dataset_slice(self, entities: List[DataSetSliceEntity]) -> bool:
        ...

    def copy_dataset(self, dataset_id: int) -> Optional[DataSetEntity]:
        ...

    def delete_dataset(self, dataset_id: int) -> int:
        ...

    def delete_dataset_slices(self, dataset_id: int, slice_ids: list) -> int:
        ...


class LabelRepository(SingleModelRepository[LabelEntity], metaclass=ABCMeta):

    def get_label_by_name(self, name: str) -> Optional[LabelEntity]:
        ...

    def update_label(self, label_id: int, label_data: dict) -> Tuple[int, str]:
        ...

    def get_labels_with_fuzzy(self, name: Optional[str]) -> List[LabelEntity]:
        ...

    def filter_labels(self, page: int, per_page: int, filters: list) -> Tuple[List[LabelEntity], dict]:
        ...

    def delete_label(self, label_id: int) -> int:
        ...


class FilterTemplateRepository(SingleModelRepository[FilterTemplateEntity], metaclass=ABCMeta):
    pass
