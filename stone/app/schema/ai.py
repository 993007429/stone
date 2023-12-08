from apiflask import Schema
from marshmallow.fields import Integer, String, Nested, Bool
from marshmallow.validate import OneOf

from stone.modules.ai.domain.enum import AIModel


class StartIn(Schema):
    slice_id = Integer(required=True)
    slice_key = String(required=True)
    ai_model = String(required=True, validate=[OneOf([member.value for member in AIModel])])
    model_version = String(required=True)


class StartOut(Schema):
    task_id = String(required=True)


class SingleStartOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(StartOut())


class PollingIn(Schema):
    task_id = String(required=True)


class PollingOut(Schema):
    done = Bool(required=True)
    rank = Integer(required=True)


class SinglePollingOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(PollingOut())
