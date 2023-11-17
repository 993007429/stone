from apiflask import Schema, PaginationSchema
from apiflask.fields import Integer, String, List, Nested, DateTime
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.schema.base import DurationField
from src.modules.slice.domain.value_objects import AiType


class StartIn(Schema):
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])
    a = String(required=True, validate=[Length(0, 255)])
    b = String(required=True, validate=[Length(0, 255)])
    c = String(required=True, validate=[Length(0, 255)])
    d = String(required=True, validate=[Length(0, 255)])


class StartOut(Schema):
    calculation_id = Integer(required=True)


class SingleStartOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(StartOut)


class PollingIn(Schema):
    calculation_id = Integer(required=True)


class PollingOut(Schema):
    calculation_id = Integer(required=True)


class SinglePollingOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(PollingOut)


class CalculationQuery(Schema):
    page = Integer(load_default=1)
    per_page = Integer(load_default=10, validate=Range(max=100))
    userid = Integer(required=True)


class CalculationIn(Schema):
    calculation_id = Integer(required=True)


class CalculationOut(Schema):
    userid = Integer(required=True)
    username = String(required=True)
    calculation_id = Integer(required=True)
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


class ListCalculationOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(CalculationOut)


class ResultIn(Schema):
    calculation_id = Integer(required=True)


class ResultOut(Schema):
    calculation_id = Integer(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])
    status = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)
    a = String(required=True, validate=[Length(0, 255)])
    b = String(required=True, validate=[Length(0, 255)])
    c = String(required=True, validate=[Length(0, 255)])
    d = String(required=True, validate=[Length(0, 255)])


class SingleResultOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ResultOut)
