from dataclasses import dataclass, field
from typing import Optional

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


@dataclass
class UserEntity(BaseEntity):
    username: str
    password_hash: str
    role: str
    id: int = field(default=None)
