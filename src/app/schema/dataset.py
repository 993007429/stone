from datetime import datetime

from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, Raw, Dict
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from marshmallow import validates, ValidationError, validates_schema

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType, Condition
from src.modules.slice.infrastructure.models import Slice, DataSet


class DataSetPageQuery(PageQuery):
    pass


class Filter(Schema):

    field = String(required=True, validate=OneOf(DataSet.__table__.columns.keys()))
    condition = String(required=True, validate=OneOf([i.value for i in list(Condition.__members__.values())]))
    value = Raw(required=True)


class DataSetFilter(Schema):
    filters = List(Nested(Filter))


class DataSetIn(Schema):
    name = String(required=True)
    remark = String(required=True, description='备注')


class DataSetOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    count = Integer(required=True)
    creator = String(required=True, description='创建人')
    remark = String(required=True, description='备注')
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=True, description='逻辑删除')

    # wsi_c = Integer(required=False, description='WSI数量')
    # patch_c = Integer(required=False, description='patch数量')
    # ROI_c = Integer(required=False, description='ROI数量')
    # labels_c = List(Nested({'label': Integer(required=True)}), required=False, description='各个标签数量')
    # anno_c = Integer(required=False, description='标注数量 二期')


class SingleDataSetOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Nested(DataSetOut))


class ListDataSetOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(DataSetOut)))
    pagination = Nested(PaginationSchema)


class DataSetIdsOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Integer(required=True), description='受影响数据集数量')
