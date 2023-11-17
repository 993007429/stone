import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.slice import StartIn, SingleStartOut, PollingIn, SinglePollingOut, CalculationIn, \
    ListCalculationOut, SingleCalculationOut, ResultIn, SingleResultOut, CalculationQuery
from src.app.schema.user import PageQuery
from src.app.service_factory import AppServiceFactory

slice_blueprint = APIBlueprint('数据模块', __name__, url_prefix='/slices')


@slice_blueprint.post('/start')
@slice_blueprint.input(StartIn, location='json')
@slice_blueprint.output(SingleStartOut)
@slice_blueprint.doc(summary='开始AI处理', security='ApiAuth')
def start(json_data):
    res = AppServiceFactory.slice_service.start(**json_data)
    return res.response


@slice_blueprint.post('/polling')
@slice_blueprint.input(PollingIn, location='json')
@slice_blueprint.output(SinglePollingOut)
@slice_blueprint.doc(summary='轮询查看AI处理是否完成', security='ApiAuth')
def polling(json_data):
    res = AppServiceFactory.slice_service.polling(**json_data)
    return res.response


@slice_blueprint.get('/calculations')
@slice_blueprint.input(CalculationQuery, location='query')
@slice_blueprint.output(ListCalculationOut)
@slice_blueprint.doc(summary='AI处理列表', security='ApiAuth')
def get_calculations(query_data):
    res = AppServiceFactory.slice_service.get_calculations(**query_data)
    return res.response


@slice_blueprint.get('/calculations/{calculation_id}')
@slice_blueprint.input(ResultIn, location='json')
@slice_blueprint.output(SingleResultOut)
@slice_blueprint.doc(summary='AI处理详情', security='ApiAuth')
def get_calculation_result(json_data):
    res = AppServiceFactory.slice_service.get_calculation_result(**json_data)
    return res.response


@slice_blueprint.delete('/calculations/{calculation_id}')
@slice_blueprint.input(CalculationIn, location='json')
@slice_blueprint.output(SingleCalculationOut)
@slice_blueprint.doc(summary='AI处理列表', security='ApiAuth')
def delete_calculation(json_data):
    res = AppServiceFactory.slice_service.delete_calculation(**json_data)
    return res.response



