import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.slice import ListSliceOut, SlicePageQuery, SingleSliceOut, SliceFilter
from src.app.service_factory import AppServiceFactory

slice_blueprint = APIBlueprint('切片', __name__, url_prefix='/slices')


@slice_blueprint.get('')
@slice_blueprint.input(SlicePageQuery, location='query')
@slice_blueprint.input(SliceFilter, location='json')
@slice_blueprint.output(ListSliceOut)
@slice_blueprint.doc(summary='切片列表', security='ApiAuth')
def get_slices(query_data, json_data):
    res = AppServiceFactory.slice_service.get_slices(**query_data, **json_data)
    return res.response


@slice_blueprint.get('/{slice_id}')
@slice_blueprint.output(SingleSliceOut)
@slice_blueprint.doc(summary='切片详情', security='ApiAuth')
def get_slice(json_data):
    res = AppServiceFactory.slice_service.get_slice(**json_data)
    return res.response


@slice_blueprint.delete('/{slice_id}')
@slice_blueprint.output(SingleSliceOut)
@slice_blueprint.doc(summary='删除切片', security='ApiAuth')
def delete_slice(json_data):
    res = AppServiceFactory.slice_service.delete_slice(**json_data)
    return res.response



