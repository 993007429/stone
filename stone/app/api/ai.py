from apiflask import APIBlueprint

from stone.app.schema.ai import StartIn, SingleStartOut, PollingIn, SinglePollingOut, APISinglePollingOut
from stone.app.service_factory import AppServiceFactory

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
@ai_blueprint.output(APISinglePollingOut)
@ai_blueprint.doc(summary='轮询查看AI处理是否完成', security='ApiAuth')
def get_task_status(json_data):
    res = AppServiceFactory.ai_service.get_task_status(**json_data)
    return res.response
