from typing import Optional, List

from pydantic import BaseModel

from stone.modules.slice.domain.entities import DataSetEntity, LabelEntity, SliceEntity
from stone.seedwork.domain.entities import BaseEntity


class SliceValueObject(SliceEntity):
    labels: List[str] = None
    label_url: str


class SliceThumbnailValueObject(SliceEntity):
    thumbnail_url: str


class AnalysisResult(BaseModel):
    id: int
    ai_model: str
    model_version: str
    ai_suggest: str


class SliceComparisonValueObject(BaseModel):
    id: int
    key: str
    name: str
    analysis_results: List[AnalysisResult]


class LabelValueObject(LabelEntity):
    count: Optional[int]


class DataSetValueObject(DataSetEntity):
    count: Optional[int]


class DatasetStatisticsValueObject(DataSetValueObject):
    annotations: list
    data_types: list
    label_names: list
