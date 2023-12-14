from stone.seedwork.domain.value_objects import BaseEnum


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
