from src.modules.slice.domain.services import SliceDomainService
from src.seedwork.application.responses import AppResponse


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService):
        self.domain_service = domain_service

    def get_slice_path(self, slice_id: int) -> AppResponse[str]:
        slide, message = self.domain_service.get_slice_by_id(slice_id)
        return AppResponse(message=message, data=slide.slice_path if slide else None)
