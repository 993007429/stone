from src.modules.dataset.domain.services import DatasetDomainService


class DatasetService(object):

    def __init__(self, domain_service: DatasetDomainService):
        self.domain_service = domain_service
