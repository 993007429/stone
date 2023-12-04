from typing import Optional

from stone.seedwork.domain.entities import BaseEntity


class UserEntity(BaseEntity):
    username: str
    password_hash: str
    role: str
    creator: Optional[str]

    class Config:
        orm_mode = True
