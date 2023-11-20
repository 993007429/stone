import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.analysis import StartIn, SingleStartOut, PollingIn, SinglePollingOut, CalculationIn, \
    ListAnalysesQuery, SingleCalculationOut, AnalysisIn, SingleAnalysisOut, AnalysesQuery
from src.app.service_factory import AppServiceFactory

analysis_blueprint = APIBlueprint('处理记录', __name__, url_prefix='/analyses')


@analysis_blueprint.post('/start')
@analysis_blueprint.input(StartIn, location='json')
@analysis_blueprint.output(SingleStartOut)
@analysis_blueprint.doc(summary='开始AI处理', security='ApiAuth')
def start(json_data):
    res = AppServiceFactory.slice_service.start(**json_data)
    return res.response


@analysis_blueprint.post('/polling')
@analysis_blueprint.input(PollingIn, location='json')
@analysis_blueprint.output(SinglePollingOut)
@analysis_blueprint.doc(summary='轮询查看AI处理是否完成', security='ApiAuth')
def polling(json_data):
    res = AppServiceFactory.slice_service.polling(**json_data)
    return res.response


@analysis_blueprint.get('')
@analysis_blueprint.input(AnalysesQuery, location='query')
@analysis_blueprint.output(ListAnalysesQuery)
@analysis_blueprint.doc(summary='AI处理记录列表', security='ApiAuth')
def get_analyses(query_data):
    res = AppServiceFactory.slice_service.get_analyses(**query_data)
    return res.response


@analysis_blueprint.get('/{analysis_id}')
@analysis_blueprint.input(AnalysisIn, location='json')
@analysis_blueprint.output(SingleAnalysisOut)
@analysis_blueprint.doc(summary='AI处理详情', security='ApiAuth')
def get_analysis(json_data):
    res = AppServiceFactory.slice_service.get_analysis(**json_data)
    return res.response


@analysis_blueprint.delete('/{analysis_id}')
@analysis_blueprint.input(CalculationIn, location='json')
@analysis_blueprint.output(SingleCalculationOut)
@analysis_blueprint.doc(summary='删除AI处理记录', security='ApiAuth')
def delete_analysis(json_data):
    res = AppServiceFactory.slice_service.delete_analysis(**json_data)
    return res.response



