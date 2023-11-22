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
    task_id = String(required=True)


class SingleStartOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(StartOut)


class PollingIn(Schema):
    task_id = String(required=True)


class PollingOut(Schema):
    analysis_id = Integer(required=True)


class SinglePollingOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(PollingOut)
