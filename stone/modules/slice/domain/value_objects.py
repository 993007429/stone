from typing import Optional

from stone.modules.slice.domain.entities import DataSetEntity, LabelEntity


class LabelValueObject(LabelEntity):
    count: Optional[int]


class DataSetValueObject(DataSetEntity):
    count: Optional[int]


class DatasetStatisticsValueObject(DataSetValueObject):
    annotations: list
    data_types: list
    label_names: list
