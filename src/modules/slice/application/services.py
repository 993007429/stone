import os
import uuid

from werkzeug.utils import secure_filename

from src.modules.slice.domain.services import SliceDomainService
from src.seedwork.application.responses import AppResponse
from src.libs.heimdall.dispatch import open_slide


class SliceService(object):

    def __init__(self, domain_service: SliceDomainService):
        self.domain_service = domain_service

    def get_slice_path(self, slice_id: int) -> AppResponse[dict]:
        slice_, message = self.domain_service.get_slice_by_id(slice_id)
        return AppResponse(message=message, data={'slice_path': slice_.slice_path if slice_ else None})

    def upload_slice(self, **kwargs) -> AppResponse[dict]:
        slice_key = self.domain_service.upload_slice(**kwargs)
        return AppResponse(data={'slice_key': slice_key})

    def create_slice(self, **kwargs) -> AppResponse[dict]:
        slice_, message = self.domain_service.create_slice(**kwargs)
        if not slice_:
            return AppResponse(err_code=1, message=message)
        return AppResponse(message=message, data={'slice': slice_.dict()})

    def filter_slices(self, **kwargs) -> AppResponse[dict]:
        slices, pagination, message = self.domain_service.filter_slices(**kwargs)
        return AppResponse(message=message, data={'slices': [slice_.dict() for slice_ in slices]}, pagination=pagination)

    def get_slice(self, slice_id: int) -> AppResponse[dict]:
        slice_, message = self.domain_service.get_slice_by_id(slice_id)
        return AppResponse(data={'slice': slice_.dict()})

    def get_label(self, label_id: int) -> AppResponse[dict]:
        label, message = self.domain_service.get_label_by_id(label_id)
        return AppResponse(data={'label': label.dict()})

    def delete_slices(self, **kwargs) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_slices(**kwargs)
        return AppResponse(message=message, data={'deleted_count': deleted_count})

    def update_slices(self, **kwargs) -> AppResponse[dict]:
        updated_count, message = self.domain_service.update_slices(**kwargs)
        return AppResponse(message=message, data={'updated_count': updated_count})

    def add_labels(self, **kwargs) -> AppResponse[dict]:
        affected_count, message = self.domain_service.add_labels(**kwargs)
        return AppResponse(message=message, data={'affected_count': affected_count})

    def create_label(self, **kwargs) -> AppResponse[dict]:
        label, message = self.domain_service.create_label(**kwargs)
        return AppResponse(message=message, data={'label': label.dict()})

    def filter_labels(self, **kwargs) -> AppResponse[dict]:
        labels, pagination, message = self.domain_service.filter_labels(**kwargs)
        return AppResponse(message=message, data={'labels': [label.dict() for label in labels]}, pagination=pagination)

    def delete_labels(self, **kwargs) -> AppResponse[dict]:
        deleted_count, message = self.domain_service.delete_labels(**kwargs)
        return AppResponse(message=message, data={'deleted_count': deleted_count})

    def update_label(self, **kwargs) -> AppResponse[dict]:
        updated_count, message = self.domain_service.update_label(**kwargs)
        return AppResponse(message=message, data={'updated_count': updated_count})
