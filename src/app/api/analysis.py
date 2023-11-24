import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.schema.analysis import AnalysesQuery, ListAnalysesOut
from src.app.service_factory import AppServiceFactory

analysis_blueprint = APIBlueprint('ai处理记录', __name__, url_prefix='/analyses')


@analysis_blueprint.get('')
@analysis_blueprint.input(AnalysesQuery, location='query')
@analysis_blueprint.output(ListAnalysesOut)
@analysis_blueprint.doc(summary='AI处理记录列表', security='ApiAuth')
def get_analyses(query_data):
    res = AppServiceFactory.ai_service.get_analyses(**query_data)
    return res.response


@analysis_blueprint.get('/{analysis_id}')
@analysis_blueprint.output(ListAnalysesOut)
@analysis_blueprint.doc(summary='AI处理详情', security='ApiAuth')
def get_analysis(analysis_id):
    res = AppServiceFactory.ai_service.get_analysis(analysis_id)
    return res.response


@analysis_blueprint.delete('/{analysis_id}')
@analysis_blueprint.output(ListAnalysesOut)
@analysis_blueprint.doc(summary='删除AI处理记录', security='ApiAuth')
def delete_analysis(analysis_id):
    res = AppServiceFactory.ai_service.delete_analysis(analysis_id)
    return res.response



