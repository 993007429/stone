from apiflask import Schema
from apiflask.fields import Integer, String
from apiflask.validators import Range
from marshmallow import ValidationError
from marshmallow.fields import Nested, Float, List


def validate_positive_integers(num):
    if not isinstance(num, int) or num <= 0:
        raise ValidationError("Positive integers only.")


class PageQuery(Schema):
    page = Integer(load_default=1, validate=Range(min=1))
    per_page = Integer(load_default=10, validate=Range(max=10000))


class NameFuzzyQuery(Schema):
    name = String(required=True, allow_none=True)


class PaginationSchema(Schema):
    page = Integer()
    per_page = Integer()
    total = Integer()


class AffectedCountOut(Schema):
    affected_count = Integer(required=True)


class APIAffectedCountOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(AffectedCountOut())


class Coordinate(Schema):
    x = List(Float())
    y = List(Float())
