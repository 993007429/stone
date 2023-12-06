from apiflask import Schema
from marshmallow.fields import Integer, String, DateTime, List, Nested
from marshmallow.validate import OneOf

from stone.app.base_schema import DurationField, PageQuery, PaginationSchema
from stone.modules.ai.domain.enum import AnalysisStat


class AnalysesQuery(PageQuery):
    userid = Integer(required=True)


class AnalysisOut(Schema):
    id = Integer(required=True)
    userid = Integer(required=True)
    username = String(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True)
    model_version = String(required=True)
    status = String(required=True, validate=[OneOf([AnalysisStat.success.name, AnalysisStat.failed.name])])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)


class ListAnalysesOut(Schema):
    analyses: List(Nested(AnalysisOut()))


class APIListAnalysesOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListAnalysesOut())
    pagination = Nested(PaginationSchema())
