from apiflask import Schema
from apiflask.fields import Integer, String, Nested, DateTime, Raw
from apiflask.validators import Range


class DurationField(Raw):
    def format(self, value):
        minutes, seconds = divmod(value, 60)
        return f'{int(minutes)}分{int(seconds)}秒'


class PageQuery(Schema):
    page = Integer(load_default=1, validate=Range(min=1))
    per_page = Integer(load_default=10, validate=Range(max=10000))


class NameFuzzyQuery(Schema):
    name = String(required=False)


class PaginationSchema(Schema):
    page = Integer()
    per_page = Integer()
    total = Integer()
