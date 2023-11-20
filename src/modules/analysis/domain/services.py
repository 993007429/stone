from src.modules.analysis.infrastructure.repositories import SQLAlchemyAnalysisRepository


class AnalysisDomainService(object):

    def __init__(self, repository: SQLAlchemyAnalysisRepository):
        self.repository = repository
