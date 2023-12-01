from apiflask import Schema
from marshmallow.fields import Integer, String, Nested, DateTime

from stone.app.base_schema import DurationField, PageQuery, PaginationSchema


class AnalysesQuery(PageQuery):
    userid = Integer(required=True)


class AnalysisOut(Schema):
    id = Integer(required=True)
    userid = Integer(required=True)
    username = String(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True)
    model_version = String(required=True)
    status = String(required=True)
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)


class ListAnalysesOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(AnalysisOut)
    pagination = Nested(PaginationSchema)
