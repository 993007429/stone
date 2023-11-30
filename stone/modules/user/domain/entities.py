from typing import Optional
from datetime import datetime

from stone.seedwork.domain.entities import BaseEntity
from stone.seedwork.domain.value_objects import GenericUUID


class UserEntity(BaseEntity):
    username: str
    password_hash: str
    role: str
    creator: str
