import asyncio
from typing import List

from apiflask import APIBlueprint, FileSchema
from flask import send_from_directory
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.slice import ListSliceOut, SlicePageQuery, SingleSliceOut, SliceFilter, SliceIdsOut, SliceIdsIn, \
    WSIIn, SliceId, ROIIn, LabelSliceIdsIn, DSSliceIdsIn, ComparisonSliceFilter, ComparisonListSliceOut
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


@slice_blueprint.get('/<int:slice_id>')
@slice_blueprint.output(SingleSliceOut)
@slice_blueprint.doc(summary='切片详情', security='ApiAuth')
def get_slice(slice_id):
    res = AppServiceFactory.slice_service.get_slice(slice_id)
    return res.response


@slice_blueprint.delete('')
@slice_blueprint.input(SliceIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='批量删除切片', security='ApiAuth')
def delete_slice(json_data):
    res = AppServiceFactory.slice_service.delete_slice(**json_data)
    return res.response


@slice_blueprint.put('')
@slice_blueprint.input(SliceIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='批量更新切片', security='ApiAuth')
def update_slice(json_data):
    res = AppServiceFactory.slice_service.update_slice(**json_data)
    return res.response


@slice_blueprint.get('/comparison')
@slice_blueprint.input(SlicePageQuery, location='query')
@slice_blueprint.input(ComparisonSliceFilter, location='json')
@slice_blueprint.output(ComparisonListSliceOut)
@slice_blueprint.doc(summary='对比模式', security='ApiAuth')
def get_comparison_slices(query_data, json_data):
    res = AppServiceFactory.slice_service.get_comparison_slices(**query_data, **json_data)
    return res.response


@slice_blueprint.put('/add-label')
@slice_blueprint.input(LabelSliceIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='添加标签', security='ApiAuth')
def add_label(json_data):
    res = AppServiceFactory.slice_service.add_label(**json_data)
    return res.response


@slice_blueprint.put('/add-ds')
@slice_blueprint.input(DSSliceIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='添加到数据集', security='ApiAuth')
def add_dataset(json_data):
    res = AppServiceFactory.slice_service.add_dataset(**json_data)
    return res.response


@slice_blueprint.get('/WSI')
@slice_blueprint.input(WSIIn, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='切片全场图', security='ApiAuth')
def get_wsi(query_data):
    res = AppServiceFactory.slice_service.get_wsi(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')


@slice_blueprint.get('/ROI')
@slice_blueprint.input(ROIIn, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='ROI', security='ApiAuth')
def get_roi(query_data):
    res = AppServiceFactory.slice_service.get_roi(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')


@slice_blueprint.get('/label')
@slice_blueprint.input(SliceId, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='切片标签', security='ApiAuth')
def get_label(query_data):
    res = AppServiceFactory.slice_service.get_label(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')









