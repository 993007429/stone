from src.modules.slice.infrastructure.repositories import SQLAlchemyDatasetRepository


class DatasetDomainService(object):

    def __init__(self, repository: SQLAlchemyDatasetRepository):
        self.repository = repository
