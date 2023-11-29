import logging
import os
import uuid
from typing import Tuple, Optional, List

from werkzeug.utils import secure_filename

import setting
from src.app.request_context import request_context
from src.infra.fs import fs
from src.infra.session import transaction
from src.libs.heimdall.dispatch import open_slide
from src.modules.slice.domain.entities import SliceEntity, LabelEntity, DataSetEntity
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

    def get_dataset_by_id(self, dataset_id: int) -> Tuple[Optional[DataSetEntity], str]:
        dataset = self.repository.get_dataset_by_id(dataset_id)
        if not dataset:
            return None, 'no dataset'
        return dataset, 'get dataset succeed'

    def get_slice_fields(self) -> list:
        fields = self.repository.get_slice_fields()
        return fields

    def upload_slice(self, **kwargs) -> Tuple[str, str]:
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

        return slice_key, slice_filename

    def create_slice(self, **kwargs) -> Tuple[Optional[SliceEntity], str]:
        slice_ = SliceEntity.parse_obj(kwargs)
        succeed, new_slice = self.repository.save_slice(slice_)
        if not succeed:
            return None, 'Duplicate slice'
        return new_slice, 'Create slice succeed'

    @transaction
    def create_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        label_exist = self.repository.get_label_by_name(kwargs['name'])
        if label_exist:
            return None, 'Duplicate label name'

        if current_user := request_context.current_user:
            username = current_user.username
            kwargs['creator'] = username

        label = LabelEntity.parse_obj(kwargs)
        succeed, new_label = self.repository.save_label(label)
        if not succeed:
            return None, 'Duplicate label'
        return new_label, 'Create label succeed'

    @transaction
    def create_dataset(self, **kwargs) -> Tuple[Optional[DataSetEntity], str]:
        kwargs['userid'] = 1
        if current_user := request_context.current_user:
            userid = current_user.userid
            username = current_user.username
            kwargs['userid'] = userid
            kwargs['creator'] = username

        dataset = DataSetEntity.parse_obj(kwargs)
        succeed, new_data_set = self.repository.save_data_set(dataset)
        if not succeed:
            return None, 'Create data set failed'
        return new_data_set, 'Create data set succeed'

    def filter_slices(self, **kwargs) -> Tuple[List[SliceEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']
        slices, pagination = self.repository.filter_slices(page, per_page, logic, filters)

        new_slices = []
        for slice_ in slices:
            slice_labels = self.repository.get_slice_labels_by_slice(slice_.id)
            slice_dict = slice_.dict()
            slice_dict['labels'] = [slice_label.label_name for slice_label in slice_labels]
            new_slice = SliceEntity.parse_obj(slice_dict)
            new_slices.append(new_slice)
        return new_slices, pagination, 'filter slices succeed'

    def filter_labels(self, **kwargs) -> Tuple[List[LabelEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']
        labels, pagination = self.repository.filter_labels(page, per_page, filters)

        new_labels = []
        for label in labels:
            slice_labels = self.repository.get_slice_labels_by_label(label.id)
            label_dict = label.dict()
            label_dict['count'] = len(slice_labels)
            new_label = LabelEntity.parse_obj(label_dict)
            new_labels.append(new_label)
        return new_labels, pagination, 'filter labels succeed'

    def filter_datasets(self, **kwargs) -> Tuple[List[DataSetEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']
        datasets, pagination = self.repository.filter_datasets(page, per_page, filters)

        new_datasets = []
        # for label in labels:
        #     slice_labels = self.repository.get_slice_labels_by_label(label.id)
        #     label_dict = label.dict()
        #     label_dict['count'] = len(slice_labels)
        #     new_label = LabelEntity.parse_obj(label_dict)
        #     new_labels.append(new_label)
        return datasets, pagination, 'filter datasets succeed'

    def get_datasets_for_user(self, **kwargs) -> Tuple[List[DataSetEntity], str]:
        # userid = request_context.current_user.userid
        userid = 1
        name = kwargs.get('name')
        datasets = self.repository.get_datasets_for_user(userid, name)
        return datasets, 'filter datasets succeed'

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

    def add_slices(self, **kwargs) -> Tuple[int, str]:
        dataset_ids = kwargs['dataset_ids']
        slice_ids = kwargs['slice_ids']
        for dataset_id in dataset_ids:
            dataset_slices = self.repository.get_dataset_slices_by_dataset(dataset_id)
            exist_slice_ids = [dataset_slice.slice_id for dataset_slice in dataset_slices]
            new_slice_ids = [slice_id for slice_id in slice_ids if slice_id not in exist_slice_ids]
            success = self.repository.add_slices(dataset_id, new_slice_ids)
            if not success:
                return 0, 'add slices to datasets failed'
        return len(dataset_ids), 'add slices to datasets succeed'

    def delete_label(self, label_id: int) -> Tuple[int, str]:
        deleted_count = self.repository.delete_label(label_id)
        return deleted_count, 'delete labels succeed'

    @transaction
    def delete_dataset(self, dataset_id: int) -> Tuple[int, str]:
        deleted_count = self.repository.delete_dataset(dataset_id)
        return deleted_count, 'delete dataset succeed'

    def update_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        label_id = kwargs['label_id']
        label_data = kwargs['label_data']
        label_exist = self.repository.get_label_by_name(label_data.get('name'))
        if label_exist:
            return None, 'Duplicate label name'

        updated_count, message = self.repository.update_label(label_id, label_data)
        new_label = self.repository.get_label_by_id(label_id)
        if updated_count:
            return new_label, message
        return None, message

    def update_dataset(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        dataset_id = kwargs['dataset_id']
        dataset_data = kwargs['dataset_data']

        updated_count, message = self.repository.update_dataset(dataset_id, dataset_data)
        new_dataset = self.repository.get_dataset_by_id(dataset_id)
        if updated_count:
            return new_dataset, message
        return None, message



