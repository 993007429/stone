from apiflask import Schema
from marshmallow.fields import Integer, String, List, Nested, Dict


class FieldCondition(Schema):
    field = String(required=True)
    condition = String(required=True)


class FilterTemplateIn(Schema):
    name = String(required=True)
    logic = String(required=True)
    fields = List(Nested(FieldCondition()), required=True)


class FilterTemplateOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    logic = String(required=True)
    fields = List(Nested(FieldCondition()), required=True)


class SingleFilterTemplateOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(FilterTemplateOut())


class ListFilterTemplateOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(FilterTemplateOut())), required=True)
