from src.modules.ai.application import tasks
from src.modules.ai.domain.services import AiDomainService
from src.seedwork.application.responses import AppResponse


class AiService(object):

    def __init__(self, domain_service: AiDomainService):
        self.domain_service = domain_service

    def start_ai_analysis(self, **kwargs) -> AppResponse:
        res = tasks.run_ai_task()
        return res

    def run_ai_task(self):
        message = self.domain_service.run_tct()
        return AppResponse(message='ai analysis succeed')
