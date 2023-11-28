import enum

from src.seedwork.domain.value_objects import BaseEnum, BaseValueObject


@enum.unique
class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


@enum.unique
class DataType(BaseEnum):
    wsi = 1  # 全场图
    roi = 2  # roi
    patch = 3  # patch
    text = 4,  # 文本


class SliceAnalysisStat(BaseEnum):
    default = 1  # 待处理
    analyzing = 2  # 处理中
    success = 3  # 已处理
    failed = 4  # 处理异常
    time_out = 5  # 处理超时
