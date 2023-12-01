from apiflask import Schema
from marshmallow.fields import Integer, String, List, Nested, DateTime, Raw, Dict, Float
from marshmallow import validates_schema, ValidationError
from marshmallow.validate import OneOf

from stone.app.base_schema import PageQuery, PaginationSchema, validate_positive_integers, AffectedCountOut
from stone.modules.slice.domain.value_objects import Condition
from stone.modules.slice.infrastructure.models import DataSet


class DataSetPageQuery(PageQuery):
    pass


class Filter(Schema):

    field = String(required=True, validate=OneOf(DataSet.__table__.columns.keys()))
    condition = String(required=True, validate=OneOf([i.value for i in list(Condition.__members__.values())]))
    value = Raw(required=True)


class DataSetFilter(Schema):
    filters = List(Nested(Filter()))


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


class SingleDataSetOut(Schema):
    dataset = Nested(DataSetOut())


class APISingleDataSetOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleDataSetOut())


class StatisticsOut(Schema):
    name = String()
    count = Integer()
    ratio = Float()


class DataSetStatisticsOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    creator = String(required=True, description='创建人')
    remark = String(required=True, description='备注')
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    annotations = List(Nested(StatisticsOut()))
    data_types = List(Nested(StatisticsOut()))
    label_names = List(Nested(StatisticsOut()))


class SingleDataSetStatisticsOut(Schema):
    dataset_statistics = Nested(DataSetStatisticsOut())


class APISingleDataSetStatisticsOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleDataSetStatisticsOut())


class ListDataSetOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(DataSetOut())))
    pagination = Nested(PaginationSchema())


class DataSetAndSliceIdsIn(Schema):
    dataset_ids = List(Integer(required=True, validate=validate_positive_integers), description='数据集ID列表', required=True)
    slice_ids = List(Integer(required=True, validate=validate_positive_integers), description='切片ID列表', required=True)

    @validates_schema(pass_many=True)
    def validate_value(self, data, **kwargs):
        dataset_ids = data.get('dataset_ids')
        slice_ids = data.get('slice_ids')

        if not dataset_ids or not slice_ids:
            raise ValidationError('List cannot be empty.')


class DSSliceIdsIn(Schema):
    ids = List(Integer(required=True), description='切片ID列表')
    dataset_id = Integer(required=True, description='数据集ID')
