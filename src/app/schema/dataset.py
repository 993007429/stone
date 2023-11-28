from datetime import datetime

from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, Raw
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from marshmallow import validates, ValidationError

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType, Condition
from src.modules.slice.infrastructure.models import Slice


class DSPageQuery(PageQuery):
    pass


class Filter(Schema):

    field = String(required=True, validate=OneOf(Slice.__table__.columns.keys()))
    condition = String(required=True, validate=OneOf([i.value for i in list(Condition.__members__.values())]))
    value = Raw(required=True)

    @validates('value')
    def validate_value(self, value):
        try:
            return datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
        except Exception:
            if isinstance(value, (bool, int, str)):
                return value
            else:
                raise ValidationError('value must be a bool or integer or str or datatime')


class DSFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class DSIn(Schema):
    name = String(required=True)
    remark = String(required=True, description='备注')


class DSOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    creator = String(required=True, description='创建人')
    version = String(required=True, description='版本号')
    remark = String(required=True, description='备注')
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=True, description='逻辑删除')

    wsi_c = Integer(required=False, description='WSI数量')
    patch_c = Integer(required=False, description='patch数量')
    ROI_c = Integer(required=False, description='ROI数量')
    labels_c = List(Nested({'label': Integer(required=True)}), required=False, description='各个标签数量')
    anno_c = Integer(required=False, description='标注数量 二期')


class SingleDSOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(DSOut)


class ListDSOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(DSOut))
    pagination = Nested(PaginationSchema)

