from src.modules.analysis.domain.services import AnalysisDomainService


class AnalysisService(object):

    def __init__(self, domain_service: AnalysisDomainService):
        self.domain_service = domain_service
