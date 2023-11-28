import logging
import os
import uuid
from typing import Tuple, Optional, List

from werkzeug.utils import secure_filename

import setting
from src.infra.fs import fs
from src.infra.session import transaction
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

    def get_label_by_id(self, label_id: int) -> Tuple[Optional[LabelEntity], str]:
        label = self.repository.get_label_by_id(label_id)
        if not label:
            return None, 'no label'
        return label, 'get label succeed'

    def upload_slice(self, **kwargs) -> str:
        slice_file = kwargs['slice_file']

        slice_filename = secure_filename(slice_file.filename)
        slice_key = uuid.uuid4().hex
        slice_dir = os.path.join(setting.DATA_DIR, slice_key)
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
        if not succeed:
            return None, 'Duplicate slice'
        return new_slice, 'Create slice succeed'

    def create_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        label = LabelEntity.parse_obj(kwargs)
        succeed, new_label = self.repository.save_label(label)
        if not succeed:
            return None, 'Duplicate label'
        return new_label, 'Create label succeed'

    def filter_slices(self, **kwargs) -> Tuple[List[SliceEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']
        slices, pagination = self.repository.filter_slices(page, per_page, logic, filters)
        return slices, pagination, 'filter slices succeed'

    def filter_labels(self, **kwargs) -> Tuple[List[LabelEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']
        labels, pagination = self.repository.filter_labels(page, per_page, filters)
        return labels, pagination, 'filter labels succeed'

    @transaction
    def delete_slices(self, **kwargs) -> Tuple[int, str]:
        slices = self.repository.get_slices(kwargs['ids'])
        deleted_count = self.repository.delete_slices(kwargs['ids'])
        for slice_ in slices:
            slice_dir = os.path.join(setting.DATA_DIR, slice_.slice_key)
            fs.remove_dir(slice_dir)
        return deleted_count, 'delete slices succeed'

    def update_slices(self, **kwargs) -> Tuple[int, str]:
        ids = kwargs.pop('ids')
        update_data = kwargs
        updated_count = self.repository.update_slices(ids, update_data)
        return updated_count, 'update slices succeed'

    def add_labels(self, **kwargs) -> Tuple[int, str]:
        slice_ids = kwargs['slice_ids']
        label_ids = kwargs['label_ids']
        affected_count = self.repository.add_labels(slice_ids, label_ids)
        return affected_count, 'add labels to slices succeed'



    def delete_labels(self, **kwargs) -> Tuple[int, str]:
        deleted_count = self.repository.delete_labels(**kwargs)
        return deleted_count, 'delete labels succeed'

    def update_label(self, **kwargs) -> Tuple[int, str]:
        updated_count, message = self.repository.update_label(**kwargs)
        if updated_count:
            return updated_count, message
        return 0, message


























