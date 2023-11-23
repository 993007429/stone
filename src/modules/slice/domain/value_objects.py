import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"
