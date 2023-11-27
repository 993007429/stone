import logging
import os
import uuid
from typing import Tuple, Optional

from werkzeug.utils import secure_filename

import setting
from src.libs.heimdall.dispatch import open_slide
from src.modules.slice.domain.entities import SliceEntity
from src.modules.slice.infrastructure.repositories import SQLAlchemySliceRepository

logger = logging.getLogger(__name__)


class SliceDomainService(object):

    def __init__(self, repository: SQLAlchemySliceRepository):
        self.repository = repository

    def get_slice_by_id(self, slice_id: int) -> Tuple[Optional[SliceEntity], str]:
        slide = self.repository.get_slice_by_id(slice_id)
        if not slide:
            return None, 'no slice'
        return slide, 'get slice success'

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
        slide = SliceEntity.parse_obj(kwargs)
        success, new_slice = self.repository.save(slide)
        if success:
            return new_slice, 'create slice success'
        return None, 'create slice failed'


























