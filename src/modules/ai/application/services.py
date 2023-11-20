from src.modules.ai.domain.services import AiDomainService


class AiService(object):

    def __init__(self, domain_service: AiDomainService):
        self.domain_service = domain_service
