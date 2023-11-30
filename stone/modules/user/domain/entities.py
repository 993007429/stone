from stone.seedwork.domain.entities import BaseEntity


class UserEntity(BaseEntity):
    username: str
    password_hash: str
    role: str
    creator: str
