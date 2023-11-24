from typing import Optional
from datetime import datetime

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


class MarkGroupEntity(BaseEntity):
    pass


class MarkEntity(BaseEntity):
    group: Optional[MarkGroupEntity] = None
    area: float = 0
    is_in_manual: bool = False
    mark_type: int

    @property
    def slice_path(self):
        return ''
