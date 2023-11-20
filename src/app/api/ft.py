import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.ft import FTIn, FTOut, SingleFTOut, ListFTOut
from src.app.service_factory import AppServiceFactory

ft_blueprint = APIBlueprint('筛选模板', __name__, url_prefix='/filter-templates')


@ft_blueprint.get('')
@ft_blueprint.output(ListFTOut)
@ft_blueprint.doc(summary='筛选模板列表', security='ApiAuth')
def get_fts(query_data, json_data):
    res = AppServiceFactory.slice_service.get_fts(**query_data, **json_data)
    return res.response


@ft_blueprint.get('/<int:ft_id>')
@ft_blueprint.output(SingleFTOut)
@ft_blueprint.doc(summary='筛选模板详情', security='ApiAuth')
def get_ft(ft_id):
    res = AppServiceFactory.slice_service.get_ft(ft_id)
    return res.response


@ft_blueprint.post('')
@ft_blueprint.input(FTIn, location='json')
@ft_blueprint.output(FTOut)
@ft_blueprint.doc(summary='保存筛选模板', security='ApiAuth')
def get_slices(query_data, json_data):
    res = AppServiceFactory.slice_service.get_slices(**query_data, **json_data)
    return res.response


@ft_blueprint.delete('/<int:ft_id>')
@ft_blueprint.output(SingleFTOut)
@ft_blueprint.doc(summary='删除筛选模板', security='ApiAuth')
def delete_ft(ft_id):
    res = AppServiceFactory.slice_service.delete_ft(ft_id)
    return res.response

















