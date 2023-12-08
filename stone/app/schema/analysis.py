from apiflask import Schema
from marshmallow.fields import Integer, String, DateTime, List, Nested, Bool, Dict, Float
from marshmallow.validate import OneOf

from stone.app.base_schema import DurationField, PageQuery, PaginationSchema, Coordinate
from stone.modules.ai.domain.enum import AnalysisStat


class AnalysesQuery(PageQuery):
    slice_id = Integer(required=True)
    userid = Integer(required=False)


class AnalysisOut(Schema):
    id = Integer(required=True)
    userid = Integer(required=True)
    username = String(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True)
    model_version = String(required=True)
    status = String(required=True, validate=[OneOf([AnalysisStat.success.value, AnalysisStat.failed.value])])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)
    delete_permission = Bool(required=True)


class ListAnalysesOut(Schema):
    analyses = List(Nested(AnalysisOut()))


class APIListAnalysesOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListAnalysesOut())
    pagination = Nested(PaginationSchema())


class QueryRoiIn(Schema):
    analysis_id = Integer(required=True, description='AI分析记录ID')
    roi_args = String(required=True, description='Roi 参数')
    roi_id = Integer(required=True, description='Roi ID')


class QueryMarksIn(Schema):
    analysis_id = Integer(required=True, description='AI分析记录ID')


class AiResult(Schema):
    index = Integer(required=True)
    neg_tumor = Integer(required=True)
    normal_cell = Integer(required=True)
    pos_tumor = Integer(required=True)
    total = Integer(required=True)
    whole_slide = Integer(required=True)


class TctAiResult(Schema):
    cell_num = Integer(required=True, allow_none=None)
    clarity = Float(required=True, allow_none=None)
    slide_quality = Integer(required=True, allow_none=None)
    diagnosis = List(String())
    microbe = List(String())
    cells = Dict()
    whole_slide = Integer(required=True, allow_none=None)


class MarkOut(Schema):
    ai_result = Nested(TctAiResult(), description='算法结果')
    area_id = String(required=True, allow_none=True, description='算法区域标注id')
    create_time = String(required=True, allow_none=True, description='')
    dashed = String(required=True, allow_none=True, description='虚线为1，实线为0')
    diagnosis = String(required=True, allow_none=True, description='')
    doctor_diagnosis = String(required=True, allow_none=True, description='医生手工判读')
    editable = Integer(required=True, allow_none=True, description='可编辑为1，不可编辑为0')
    fill_color = String(required=True, allow_none=True, description='填充颜色')
    group_id = Integer(required=True, allow_none=True, description='标注组id')
    id = String(required=True, allow_none=True, description='')
    image = String(required=True, allow_none=True, description='')
    mark_type = Integer(required=True, allow_none=True, description='标注类型（手动标注1、算法标注2、算法标注区域3）')
    method = String(required=True, allow_none=True, description='AI分析记录ID')
    path = Nested(Coordinate(), description='AI分析记录ID')
    position = Nested(Coordinate(), description='AI分析记录ID')
    radius = Integer(required=True, allow_none=True, description='标注直径')
    remark = String(required=True, allow_none=True, description='AI分析记录ID')
    show_layer = Integer(required=True, allow_none=True, description='AI分析记录ID')
    stroke_color = String(required=True, allow_none=True, description='AI分析记录ID')


class ListMarkOut(Schema):
    marks = List(Nested(MarkOut()))


class APIListMarkOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListMarkOut())
