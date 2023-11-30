from typing import TypeVar, Generic

from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, URL, Dict
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from stone.app.base_schema import DurationField, PageQuery, PaginationSchema
from stone.modules.slice.domain.value_objects import LogicType


class FieldCondition(Schema):
    field = String(required=True)
    condition = String(required=True)


class FTIn(Schema):
    name = String(required=True)
    logic = String(required=True)
    fields = List(Nested(FieldCondition()))


class FTOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    logic = String(required=True)
    fields = List(Nested(FieldCondition()), required=True)


class SingleFTOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Nested(FTOut()), required=True)


class ListFTOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(FTOut())), required=True)










