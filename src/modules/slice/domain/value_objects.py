import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


@enum.unique
class WhStat(BaseEnum):
    NOT_IN_STOCK = 0  # 待入库
    IN_STOCK = 1  # 已入库


@enum.unique
class DataType(BaseEnum):
    WSI = 'WSI'  # 全场图
    ROI = 'ROI'  # roi
    PATCH = 'PATCH'  # patch
    TEXT = 'TEXT',  # 文本
