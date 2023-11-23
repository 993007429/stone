from typing import Optional
from datetime import datetime

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


class SliceEntity(BaseEntity):
    username: str
    password_hash: str
    role: str
    creator: str

    @property
    def slice_path(self):
        return ''
