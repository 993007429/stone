from apiflask import Schema
from marshmallow.fields import Integer, String, List, Nested, Dict


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
