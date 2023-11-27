from typing import Optional
from datetime import datetime

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


class SliceEntity(BaseEntity):
    slice_key: Optional[str]
    parent_id: Optional[int]
    name: str
    data_type: str
    wh_stat: str
    is_deleted: int = 0

    class Config:
        orm_mode = True

    @property
    def slice_path(self):
        return ''
