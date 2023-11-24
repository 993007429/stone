import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.schema.ai import StartIn, SingleStartOut, PollingIn, SinglePollingOut
from src.app.service_factory import AppServiceFactory

ai_blueprint = APIBlueprint('ai', __name__, url_prefix='/ai')


@ai_blueprint.post('/start')
@ai_blueprint.input(StartIn, location='json')
@ai_blueprint.output(SingleStartOut)
@ai_blueprint.doc(summary='开始AI处理', security='ApiAuth')
def start_ai_analysis(json_data):
    res = AppServiceFactory.ai_service.start_ai_analysis(**json_data)
    return res.response


@ai_blueprint.post('/polling/task-status')
@ai_blueprint.input(PollingIn, location='json')
@ai_blueprint.output(SinglePollingOut)
@ai_blueprint.doc(summary='轮询查看AI处理是否完成', security='ApiAuth')
def polling(json_data):
    res = AppServiceFactory.ai_service.polling(**json_data)
    return res.response
