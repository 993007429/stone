import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


@enum.unique
class AiType(BaseEnum):
    admin = "admin"
    user = "user"


class AITaskVO(BaseValueObject):
    slice_id: int
    slide_path: str
    ai_model: str
    model_version: str

