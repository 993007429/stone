from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, URL
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.base_schema import DurationField, PageQuery, Filter, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType


class FieldCondition(Schema):
    field = String(required=True)
    condition = String(required=True)


class FTIn(Schema):
    name = String(required=True, validate=[Length(0, 20)])


class FTOut(Schema):
    id = Integer(required=True)
    name = String(required=True, validate=[Length(0, 20)])
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    field_condition = List(Nested(FieldCondition))


class SingleFTOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(FTOut)


class ListFTOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested({'id': Integer(required=True), 'name': String(required=True, validate=[Length(0, 20)])}))

