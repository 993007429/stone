import enum
from dataclasses import dataclass

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class RoleType(BaseEnum):
    admin = "admin"
    user = "user"


@dataclass(frozen=True)
class TokenVO(BaseValueObject):
    user_id: int
    username: str
    role: str
    token: str
