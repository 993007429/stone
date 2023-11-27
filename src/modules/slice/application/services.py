import os
import uuid

from werkzeug.utils import secure_filename

from src.modules.slice.domain.services import SliceDomainService
from src.seedwork.application.responses import AppResponse
from src.libs.heimdall.dispatch import open_slide


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService):
        self.domain_service = domain_service

    def get_slice_path(self, slice_id: int) -> AppResponse[str]:
        slide, message = self.domain_service.get_slice_by_id(slice_id)
        return AppResponse(message=message, data=slide.slice_path if slide else None)

    def upload_slice(self, **kwargs) -> AppResponse[dict]:
        slice_key = self.domain_service.upload_slice(**kwargs)
        return AppResponse(data={'slice_key': slice_key})

    def create_slice(self, **kwargs) -> AppResponse[dict]:
        slice_, message = self.domain_service.create_slice(**kwargs)
        return AppResponse(message=message, data={'slice': slice_.dict()})

    def filter_slices(self, **kwargs) -> AppResponse[dict]:
        slices, pagination, message = self.domain_service.filter_slices(**kwargs)
        return AppResponse(message=message, data={'slices': [slice_.dict() for slice_ in slices]}, pagination=pagination)

