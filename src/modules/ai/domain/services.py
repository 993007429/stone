from src.modules.ai.infrastructure.repositories import SQLAlchemyAiRepository


class AiDomainService(object):

    def __init__(self, repository: SQLAlchemyAiRepository):
        self.repository = repository
