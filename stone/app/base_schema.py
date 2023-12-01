from apiflask import Schema
from apiflask.fields import Integer, String, Raw
from apiflask.validators import Range
from marshmallow import ValidationError


class DurationField(Raw):
    def format(self, value):
        minutes, seconds = divmod(value, 60)
        return f'{int(minutes)}分{int(seconds)}秒'


class PageQuery(Schema):
    page = Integer(load_default=1, validate=Range(min=1))
    per_page = Integer(load_default=10, validate=Range(max=10000))


class NameFuzzyQuery(Schema):
    name = String(required=True, allow_none=True)


class PaginationSchema(Schema):
    page = Integer()
    per_page = Integer()
    total = Integer()


def validate_positive_integers(num):
    if not isinstance(num, int) or num <= 0:
        raise ValidationError("Positive integers only.")
