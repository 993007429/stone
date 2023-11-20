from apiflask import Schema
from apiflask.fields import Integer, String, Nested, DateTime
from apiflask.validators import Length, OneOf

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.slice.domain.value_objects import AiType


class StartIn(Schema):
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])


class StartOut(Schema):
    analysis_id = Integer(required=True)


class SingleStartOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(StartOut)


class PollingIn(Schema):
    analysis_id = Integer(required=True)


class PollingOut(Schema):
    analysis_id = Integer(required=True)


class SinglePollingOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(PollingOut)


class AnalysesQuery(PageQuery):
    userid = Integer(required=True)


class CalculationIn(Schema):
    analysis_id = Integer(required=True)


class CalculationOut(Schema):
    userid = Integer(required=True)
    username = String(required=True)
    analysis_id = Integer(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])
    status = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)


class SingleCalculationOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(CalculationOut)


class ListAnalysesQuery(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(CalculationOut)
    pagination = Nested(PaginationSchema)


class AnalysisIn(Schema):
    analysis_id = Integer(required=True)


class ResultOut(Schema):
    analysis_id = Integer(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])
    status = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)


class SingleAnalysisOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ResultOut)
