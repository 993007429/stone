from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, URL
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.base_schema import DurationField, PageQuery, Filter, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType


class LabelPageQuery(PageQuery):
    pass


class LabelFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class LabelIn(Schema):
    name = String(required=True)


class LabelOut(Schema):
    id = Integer(required=True)
    name = String(required=True, validate=[Length(0, 255)])
    count = Integer(required=True, description='数据量')
    creator = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=True)


class SingleLabelOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(LabelOut)


class ListLabelOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(LabelOut))
    pagination = Nested(PaginationSchema)







