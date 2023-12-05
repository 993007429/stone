from stone.seedwork.domain.value_objects import BaseEnum


class AnalysisStatus(BaseEnum):
    success = 1  # 已处理
    failed = 2  # 处理异常
