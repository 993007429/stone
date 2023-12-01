from stone.modules.slice.domain.entities import DataSetEntity
from stone.seedwork.domain.value_objects import BaseEnum, BaseValueObject


class LogicType(BaseEnum):
    and_ = "and"
    or_ = "or"


class DataType(BaseEnum):
    wsi = 1  # 全场图
    patch = 2  # patch
    roi = 3  # roi
    text = 4,  # 文本


class SliceAnalysisStat(BaseEnum):
    default = 1  # 待处理
    analyzing = 2  # 处理中
    success = 3  # 已处理
    failed = 4  # 处理异常
    time_out = 5  # 处理超时


class Condition(BaseEnum):
    equal = 'equal'
    unequal = 'unequal'
    greater_than = 'greater_than'
    less_than = 'less_than'
    is_null = 'is_null'
    not_null = 'not_null'
    contain = 'contain'
    not_contain = 'not_contain'


class DatasetStatisticsVO(BaseValueObject, DataSetEntity):
    annotations: list
    data_types: list
    label_names: list
