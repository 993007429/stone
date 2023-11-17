from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.base_schema import DurationField, PageQuery
from src.modules.slice.domain.value_objects import LogicType, AiType


class SlicePageQuery(PageQuery):
    pass


class Filter(Schema):
    field = String(required=True)
    condition = String(required=True)
    value = String(required=True)


class SliceFilter(Schema):
    logic = String(required=True, validate=[OneOf([LogicType.and_.value, LogicType.or_.value])])
    filters = List(Nested(Filter))


class SliceOut(Schema):
    userid = Integer(required=True)
    username = String(required=True)
    analysis_id = Integer(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True, validate=[OneOf([AiType.admin.value, AiType.user.value])])
    model_version = String(required=True, validate=[Length(8, 32)])
    status = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)


class SingleSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SliceOut)


class ListSliceOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(SliceOut))
