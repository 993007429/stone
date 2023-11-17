from src.modules.slice.domain.services import SliceDomainService


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService):
        self.domain_service = domain_service
