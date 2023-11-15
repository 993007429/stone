from dataclasses import dataclass

from src.seedwork.domain.entities import BaseEntity
from src.seedwork.domain.value_objects import GenericUUID


@dataclass
class UserEntity(BaseEntity):
    # id: GenericUUID
    username: str
    password_hash: str
    role: str

    @classmethod
    def create_user(cls, username: str, password: str, role: str):
        password_hash = password + '加密'
        return cls(username=username,
                   password_hash=password_hash,
                   role=role)
