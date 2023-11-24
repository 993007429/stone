from apiflask import Schema
from apiflask.fields import Integer, String, Nested, DateTime, List
from apiflask.validators import Length, OneOf
from marshmallow.fields import Bool

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.ai.domain.value_objects import AIType


class StartIn(Schema):
    slice_id = Integer(required=True)
    ai_model = String(required=True)
    model_version = String(required=True)


class StartOut(Schema):
    task_id = String(required=True)


class SingleStartOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(StartOut)


class PollingIn(Schema):
    task_id = String(required=True)


class PollingOut(Schema):
    done = Bool(required=True)
    rank = Integer(required=True)


class SinglePollingOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(PollingOut)
