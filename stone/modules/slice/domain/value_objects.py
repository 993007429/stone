from typing import Optional, List

from stone.modules.slice.domain.entities import DataSetEntity, LabelEntity, SliceEntity


class SliceValueObject(SliceEntity):
    labels: List[str] = None


class SliceThumbnailValueObject(SliceEntity):
    thumbnail: str


class LabelValueObject(LabelEntity):
    count: Optional[int]


class DataSetValueObject(DataSetEntity):
    count: Optional[int]


class DatasetStatisticsValueObject(DataSetValueObject):
    annotations: list
    data_types: list
    label_names: list
