from apiflask import Schema
from marshmallow.fields import Integer, String, List, Nested


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
    filter_template = Nested(FilterTemplateOut())


class ApiSingleFilterTemplateOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleFilterTemplateOut())


class ListFilterTemplateOut(Schema):
    filter_templates = List(Nested(FilterTemplateOut()))


class ApiListFilterTemplateOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListFilterTemplateOut())
