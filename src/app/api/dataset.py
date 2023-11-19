import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.dataset import DSPageQuery, DSFilter, ListDSOut, SingleDSOut, DatasetsIn, DatasetsOut, DSIn
from src.app.service_factory import AppServiceFactory

ds_blueprint = APIBlueprint('数据集', __name__, url_prefix='/datasets')


@ds_blueprint.get('')
@ds_blueprint.input(DSPageQuery, location='query')
@ds_blueprint.input(DSFilter, location='json')
@ds_blueprint.output(ListDSOut)
@ds_blueprint.doc(summary='数据集列表', security='ApiAuth')
def get_datasets(query_data, json_data):
    res = AppServiceFactory.dataset_service.get_datasets(**query_data, **json_data)
    return res.response


@ds_blueprint.get('/<int:ds_id>')
@ds_blueprint.output(SingleDSOut)
@ds_blueprint.doc(summary='数据集详情', security='ApiAuth')
def get_dataset(ds_id):
    res = AppServiceFactory.dataset_service.get_dataset(ds_id)
    return res.response


@ds_blueprint.delete('/<int:ds_id>')
@ds_blueprint.input(DSIn)
@ds_blueprint.output(SingleDSOut)
@ds_blueprint.doc(summary='更新数据集', security='ApiAuth')
def delete_dataset(ds_id):
    res = AppServiceFactory.dataset_service.delete_dataset(ds_id)
    return res.response


@ds_blueprint.delete('/<int:ds_id>')
@ds_blueprint.output(SingleDSOut)
@ds_blueprint.doc(summary='删除数据集', security='ApiAuth')
def delete_dataset(ds_id):
    res = AppServiceFactory.dataset_service.delete_dataset(ds_id)
    return res.response


