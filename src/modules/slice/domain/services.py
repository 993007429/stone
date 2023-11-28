import logging
import os
import uuid
from typing import Tuple, Optional, List

from werkzeug.utils import secure_filename

import setting
from src.libs.heimdall.dispatch import open_slide
from src.modules.slice.domain.entities import SliceEntity, LabelEntity
from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository

logger = logging.getLogger(__name__)


class SliceDomainService(object):

    def __init__(self, repository: SQLAlchemySliceRepository):
        self.repository = repository

    def get_slice_by_id(self, slice_id: int) -> Tuple[Optional[SliceEntity], str]:
        slide = self.repository.get_slice_by_id(slice_id)
        if not slide:
            return None, 'no slice'
        return slide, 'get slice succeed'

    def upload_slice(self, **kwargs) -> str:
        slice_file = kwargs['slice_file']

        slice_filename = secure_filename(slice_file.filename)
        slice_key = uuid.uuid4().hex
        slice_dir = os.path.join('D:\\data\\111', slice_key)
        slice_path = os.path.join(slice_dir, slice_filename)

        if not os.path.exists(slice_dir):
            os.makedirs(slice_dir)

        slice_file.save(slice_path)

        slide = open_slide(slice_path)

        try:
            slide.save_label(os.path.join(slice_dir, "label.png"))
            thumbnail_image = slide.get_thumbnail(setting.THUMBNAIL_BOUNDING)
            thumbnail_image.save(os.path.join(slice_dir, "thumbnail.jpeg"))
        except Exception as e:
            logger.exception(e)

        return slice_key

    def create_slice(self, **kwargs) -> Tuple[Optional[SliceEntity], str]:
        slice_ = SliceEntity.parse_obj(kwargs)
        succeed, new_slice = self.repository.save_slice(slice_)
        if succeed:
            return new_slice, 'create slice succeed'
        return None, 'create slice failed'

    def filter_slices(self, **kwargs) -> Tuple[List[SliceEntity], dict, str]:
        slices, pagination = self.repository.filter_slices(**kwargs)
        return slices, pagination, 'filter slices succeed'

    def delete_slices(self, **kwargs) -> Tuple[int, str]:
        deleted_count = self.repository.delete_slices(**kwargs)
        return deleted_count, 'delete slices succeed'

    def update_slices(self, **kwargs) -> Tuple[int, str]:
        updated_count = self.repository.update_slices(**kwargs)
        return updated_count, 'update slices succeed'

    def add_labels(self, **kwargs) -> Tuple[int, str]:
        affected_count = self.repository.add_labels(**kwargs)
        return affected_count, 'add labels to slices succeed'

    def create_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        label = LabelEntity.parse_obj(kwargs)
        succeed, new_label = self.repository.save_label(label)
        if succeed:
            return new_label, 'create label succeed'
        return None, 'create label failed'

    def filter_labels(self, **kwargs) -> Tuple[List[LabelEntity], dict, str]:
        labels, pagination = self.repository.filter_labels(**kwargs)
        return labels, pagination, 'filter labels succeed'

    def delete_labels(self, **kwargs) -> Tuple[int, str]:
        deleted_count = self.repository.delete_labels(**kwargs)
        return deleted_count, 'delete labels succeed'

    def update_labels(self, **kwargs) -> Tuple[int, str]:
        updated_count = self.repository.update_labels(**kwargs)
        return updated_count, 'update labels succeed'


























