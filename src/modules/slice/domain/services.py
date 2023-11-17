from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository


class SliceDomainService(object):

    def __init__(self, repository: SQLAlchemySliceRepository):
        self.repository = repository
