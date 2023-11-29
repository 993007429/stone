from datetime import datetime

from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime, Raw, Dict
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from marshmallow import validates, ValidationError, validates_schema

from src.app.base_schema import DurationField, PageQuery, PaginationSchema
from src.modules.slice.domain.value_objects import LogicType, Condition
from src.modules.slice.infrastructure.models import Label

columns_with_types = {column_name: str(column.type) for column_name, column in Label.__table__.columns.items()}

bool_fields, float_fields, int_fields, datetime_fields, str_fields = [], [], [], [], []
for column_name, column_type in columns_with_types.items():
    if column_type in ['BOOLEAN']:
        bool_fields.append(column_name)
    elif column_type in ['FLOAT']:
        float_fields.append(column_name)
    elif column_type in ['INTEGER', 'BIGINT', 'SMALLINT']:
        int_fields.append(column_name)
    elif column_type in ['DATETIME']:
        datetime_fields.append(column_name)
    elif column_type.startswith('VARCHAR'):
        str_fields.append(column_name)


class LabelPageQuery(PageQuery):
    pass


class Filter(Schema):

    field = String(required=True, validate=OneOf(Label.__table__.columns.keys()))
    condition = String(required=True, validate=OneOf([i.value for i in list(Condition.__members__.values())]))
    value = Raw(required=True)

    @validates_schema(pass_many=True, pass_original=True)
    def validate_value(self, data, raw_data, **kwargs):
        filed = data.get('field')
        value = data.get('value')

        if filed in bool_fields:
            if not isinstance(value, bool):
                raise ValidationError('value must be a bool')
        elif filed in float_fields:
            if not isinstance(value, float):
                raise ValidationError('value must be a float')
        elif filed in int_fields:
            if not isinstance(value, int):
                raise ValidationError('value must be an int')
        elif filed in datetime_fields:
            try:
                data['value'] = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
            except Exception:
                raise ValidationError('value must be a datatime')
        elif filed in str_fields:
            if not isinstance(value, str):
                raise ValidationError('value must be a str')


class LabelFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class LabelIn(Schema):
    name = String(required=True)


class LabelOut(Schema):
    id = Integer(required=True)
    name = String(required=True)
    count = Integer(required=False, description='数据量')
    creator = String(required=False)
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    is_deleted = Integer(required=True)


class SingleLabelOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=Nested(LabelOut))


class ListLabelOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Dict(keys=String(), values=List(Nested(LabelOut)))
    pagination = Nested(PaginationSchema)















