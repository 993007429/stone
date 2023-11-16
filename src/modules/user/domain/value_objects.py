import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class RoleType(BaseEnum):
    admin = "admin"
    user = "user"


class LoginInfo(BaseValueObject):
    userid: int
    username: str
    role: str
    token: str
