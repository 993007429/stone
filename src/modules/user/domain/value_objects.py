import enum

from src.seedwork.domain.value_objects import BaseEnum


@enum.unique
class RoleType(BaseEnum):
    admin = "admin"
    user = "user"
