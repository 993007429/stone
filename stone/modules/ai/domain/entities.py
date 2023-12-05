from typing import Optional
from pydantic import Field

from stone.seedwork.domain.entities import BaseEntity


class MarkGroupEntity(BaseEntity):
    pass


class MarkEntity(BaseEntity):
    group: Optional[MarkGroupEntity] = Field(None, description='')
    is_in_manual: bool = Field(None, description='')
    position: Optional[dict] = Field(None, description='标注点位置')
    method: Optional[str] = Field(None, description='标注工具，例如自由笔')
    is_export: Optional[int] = Field(None, description='导出到报告1，否则为0')
    remark: Optional[str] = Field(None, description='判读结果')
    ai_result: Optional[dict] = Field(None, description='算法结果')
    editable: Optional[int] = Field(None, description='可编辑为1，不可编辑为0')
    stroke_color: Optional[str] = Field(None, description='边框颜色')
    fill_color: Optional[str] = Field(None, description='填充颜色')
    mark_type: Optional[int] = Field(None, description='标注类型（手动标注1、算法标注2、算法标注区域3）')
    diagnosis: Optional[dict] = Field(None, description='')
    radius: Optional[float] = Field(None, description='标注直径')
    group_id: Optional[int] = Field(None, description='标注组id')
    area_id: Optional[int] = Field(None, description='算法区域标注id')
    dashed: Optional[int] = Field(None, description='虚线为1，实线为0')
    doctor_diagnosis: Optional[dict] = Field(None, description='医生手工判读')
    create_time: Optional[float] = Field(None, description='')

    @property
    def slice_path(self):
        return ''


class AnalysisEntity(BaseEntity):
    userid: int
    username: str
    slice_id: int
    ai_model: str
    model_version: str
    status: str
    time_consume: float
