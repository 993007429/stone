import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class AiType(BaseEnum):
    admin = "admin"
    user = "user"
