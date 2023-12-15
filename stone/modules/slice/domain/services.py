import logging
import os
import uuid
from collections import Counter
from typing import Tuple, Optional, List

import setting
from stone.app.request_context import request_context
from stone.infra.fs import fs
from stone.libs.heimdall.dispatch import open_slide
from stone.modules.slice.domain.entities import SliceEntity, LabelEntity, DataSetEntity, DataSetSliceEntity, \
    FilterTemplateEntity, AnalysisVO, AnalysisEntity
from stone.modules.slice.domain.enum import DataType
from stone.modules.slice.domain.repositories import FilterTemplateRepository, DataSetRepository, SliceRepository, \
    LabelRepository, AnalysisRepository
from stone.modules.slice.domain.value_objects import DatasetStatisticsValueObject, LabelValueObject, SliceValueObject, \
    SliceThumbnailValueObject, SliceComparisonValueObject
from stone.utils.get_path import get_slice_dir, get_tile_dir, get_slice_path, get_tile_path, get_db_dir

logger = logging.getLogger(__name__)


class SliceDomainService(object):

    def __init__(
            self,
            slice_repository: SliceRepository,
            analysis_repository: AnalysisRepository,
            dataset_repository: DataSetRepository,
            label_repository: LabelRepository,
            filter_template_repository: FilterTemplateRepository
    ):
        self.slice_repository = slice_repository
        self.analysis_repository = analysis_repository
        self.dataset_repository = dataset_repository
        self.label_repository = label_repository
        self.filter_template_repository = filter_template_repository

    def upload_slice(self, **kwargs) -> Tuple[str, str]:
        slice_file = kwargs['slice_file']

        slice_filename = slice_file.filename
        slice_key = uuid.uuid4().hex
        slice_dir = get_slice_dir(slice_key)
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
        slice_to_save = SliceEntity.parse_obj(kwargs)
        new_slice = self.slice_repository.save(slice_to_save)
        if not new_slice:
            return None, 'Create slice failed'
        return new_slice, 'Create slice succeed'

    def get_slice_fields(self) -> list:
        return self.slice_repository.get_slice_fields()

    def get_slice(self, slice_id: int) -> Tuple[Optional[SliceEntity], str]:
        slice_ = self.slice_repository.get(slice_id)
        if not slice_:
            return None, 'No slice'
        return slice_, 'Get slice succeed'

    def filter_slices(self, **kwargs) -> Tuple[List[SliceValueObject], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']
        label_ids = kwargs['filter']['label_ids']

        slice_ids = []
        if label_ids:
            slice_labels = self.slice_repository.get_slice_labels_by_label_ids(label_ids)
            slice_ids = [slice_label.slice_id for slice_label in slice_labels]

        slices, pagination = self.slice_repository.filter(page, per_page, filters, logic, set(slice_ids))

        new_slices = []
        for slice_ in slices:
            slice_labels = self.slice_repository.get_slice_labels_by_slice(slice_.id)
            slice_dict = slice_.dict()
            slice_dict['labels'] = [slice_label.label_name for slice_label in slice_labels]
            slice_dict['label_url'] = request_context.host_url + f'label/{slice_.key}'
            new_slice = SliceValueObject.parse_obj(slice_dict)
            new_slices.append(new_slice)
        return new_slices, pagination, 'Filter slices succeed'

    def filter_slice_thumbnails(self, **kwargs) -> Tuple[List[SliceEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']
        label_ids = kwargs['filter'].get('label_ids')

        slice_ids = []
        if label_ids:
            slice_labels = self.slice_repository.get_slice_labels_by_label_ids(label_ids)
            slice_ids = [slice_label.slice_id for slice_label in slice_labels]

        slices, pagination = self.slice_repository.filter(page, per_page, filters, logic, set(slice_ids))

        new_slices = []
        for slice_ in slices:
            slice_dict = slice_.dict()
            slice_dict['thumbnail_url'] = request_context.host_url + f'thumbnail/{slice_.key}'
            new_slice = SliceThumbnailValueObject.parse_obj(slice_dict)
            new_slices.append(new_slice)
        return new_slices, pagination, 'Filter slices succeed'

    def filter_comparison_slices(self, **kwargs) -> Tuple[List[SliceComparisonValueObject], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        logic = kwargs['filter']['logic']
        filters = kwargs['filter']['filters']
        ai_model = kwargs['filter']['ai_model']
        model_versions = kwargs['filter']['model_versions']

        slices, pagination = self.slice_repository.filter(page, per_page, filters, logic)

        slice_ids = [slice_.id for slice_ in slices]
        analyses_all = self.analysis_repository.gets_for_comparison(slice_ids, ai_model, model_versions)

        new_slices = []
        for slice_ in slices:
            slice_dict = slice_.dict(include={'id', 'key', 'name'})
            analyses = [analysis for analysis in analyses_all if analysis.slice_id == slice_.id]
            analysis_results = [analysis.dict(include={'id', 'ai_model', 'model_version', 'ai_suggest'}) for analysis in analyses]
            slice_dict['analysis_results'] = analysis_results
            new_slice = SliceComparisonValueObject.parse_obj(slice_dict)
            new_slices.append(new_slice)
        return new_slices, pagination, 'Filter succeed'

    def update_slices(self, **kwargs) -> Tuple[int, str]:
        ids = kwargs.pop('ids')
        update_data = kwargs
        updated_count = self.slice_repository.batch_update(ids, update_data)
        return updated_count, 'Update slices succeed'

    def add_labels(self, **kwargs) -> Tuple[int, str]:
        slice_ids = kwargs['slice_ids']
        label_ids = kwargs['label_ids']
        affected_count = self.slice_repository.add_labels(slice_ids, label_ids)
        return affected_count, 'Add labels to slices succeed'

    def delete_slices(self, **kwargs) -> Tuple[int, str]:
        slices = self.slice_repository.gets(kwargs['ids'])
        deleted_count = self.slice_repository.batch_delete(kwargs['ids'])
        for slice_ in slices:
            slice_dir = get_slice_dir(slice_.key)
            fs.remove_dir(slice_dir)
        return deleted_count, 'Delete slices succeed'

    def remove_slices(self, **kwargs) -> Tuple[int, str]:
        dataset_id = kwargs['dataset_id']
        slice_ids = kwargs['slice_ids']
        deleted_count = self.dataset_repository.delete_dataset_slices(dataset_id, slice_ids)
        return deleted_count, 'Remove slices from dataset succeed'

    def create_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        if current_user := request_context.current_user:
            username = current_user.username
            kwargs['creator'] = username
        label_to_save = LabelEntity.parse_obj(kwargs)
        new_label = self.label_repository.save(label_to_save)
        if not new_label:
            return None, 'Create label failed'
        return new_label, 'Create label succeed'

    def get_label(self, label_id: int) -> Tuple[Optional[LabelEntity], str]:
        label = self.label_repository.get(label_id)
        if not label:
            return None, 'No label'
        return label, 'Get label succeed'

    def filter_labels(self, **kwargs) -> Tuple[List[LabelValueObject], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']
        labels, pagination = self.label_repository.filter(page, per_page, filters)
        new_labels = []
        for label in labels:
            slice_labels = self.slice_repository.get_slice_labels_by_label(label.id)
            label_dict = label.dict()
            label_dict['count'] = len(slice_labels)
            new_label = LabelValueObject.parse_obj(label_dict)
            new_labels.append(new_label)
        return new_labels, pagination, 'Filter labels succeed'

    def get_labels_with_fuzzy(self, **kwargs) -> Tuple[List[LabelEntity], str]:
        name = kwargs.get('name')
        labels = self.label_repository.get_labels_with_fuzzy(name)
        return labels, 'Get labels succeed'

    def update_label(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        label_id = kwargs['label_id']
        label_data = kwargs['label_data']
        updated_count = self.label_repository.update(label_id, label_data)
        if not updated_count:
            return None, 'Update label failed'
        label = self.label_repository.get(label_id)
        return label, 'Update label succeed'

    def delete_label(self, label_id: int) -> Tuple[int, str]:
        deleted_count = self.label_repository.delete(label_id)
        if not deleted_count:
            return deleted_count, 'No label deleted'
        return deleted_count, 'Delete label succeed'

    def create_dataset(self, **kwargs) -> Tuple[Optional[DataSetEntity], str]:
        kwargs['userid'] = 1
        if current_user := request_context.current_user:
            userid = current_user.userid
            username = current_user.username
            kwargs['userid'] = userid
            kwargs['creator'] = username
        dataset_to_save = DataSetEntity.parse_obj(kwargs)
        new_data_set = self.dataset_repository.save(dataset_to_save)
        if not new_data_set:
            return None, 'Create dataset failed'
        return new_data_set, 'Create dataset succeed'

    def get_dataset(self, dataset_id: int) -> Tuple[Optional[DataSetEntity], str]:
        dataset = self.dataset_repository.get(dataset_id)
        if not dataset:
            return None, 'No dataset'
        return dataset, 'Get dataset succeed'

    def get_filter_template(self, filter_template_id: int) -> Tuple[Optional[FilterTemplateEntity], str]:
        filter_template = self.filter_template_repository.get(filter_template_id)
        if not filter_template:
            return None, 'No dataset'
        return filter_template, 'Get dataset succeed'

    def get_dataset_statistics(self, dataset_id: int) -> Tuple[Optional[DatasetStatisticsValueObject], str]:
        dataset = self.dataset_repository.get(dataset_id)
        if not dataset:
            return None, 'No dataset'

        dataset_slices = self.dataset_repository.get_dataset_slices_by_dataset(dataset_id)
        slice_ids = [dataset_slice.slice_id for dataset_slice in dataset_slices]

        annotations_count = []

        slices = self.slice_repository.gets(set(slice_ids))
        data_type_values = [slice_.data_type for slice_ in slices]
        total_data_types = len(data_type_values)

        data_type_names = [DataType.wsi.name, DataType.patch.name, DataType.roi.name, 'other']
        for data_type_value in data_type_values:
            if data_type_value == DataType.wsi.value:
                data_type_names.append(DataType.wsi.name)
            elif data_type_value == DataType.patch.value:
                data_type_names.append(DataType.patch.name)
            elif data_type_value == DataType.roi.value:
                data_type_names.append(DataType.roi.name)
            else:
                data_type_names.append('other')

        data_types_statistics = [{'name': key, 'count': count - 1, 'ratio': round((count - 1) / total_data_types, 2)}
                                 for key, count in Counter(data_type_names).items()]

        slice_labels = self.slice_repository.get_slice_labels_by_slice_ids(set(slice_ids))
        label_names = [slice_label.label_name for slice_label in slice_labels]
        total_label_names = len(label_names)
        label_names_statistics = [{'name': key, 'count': count, 'ratio': round(count / total_label_names, 2)}
                                  for key, count in Counter(label_names).items()]

        dataset_statistics = DatasetStatisticsValueObject.parse_obj(
            {
                **dataset.dict(),
                **{
                    'annotations': annotations_count,
                    'data_types': data_types_statistics,
                    'label_names': label_names_statistics
                }
            }
        )

        return dataset_statistics, 'Get dataset_statistics succeed'

    def filter_datasets(self, **kwargs) -> Tuple[List[DataSetEntity], dict, str]:
        page = kwargs['page_query']['page']
        per_page = kwargs['page_query']['per_page']
        filters = kwargs['filter']['filters']
        datasets, pagination = self.dataset_repository.filter(page, per_page, filters)

        new_datasets = []
        for dataset in datasets:
            dataset_slices = self.dataset_repository.get_dataset_slices_by_dataset(dataset.id)
            dataset_dict = dataset.dict()
            dataset_dict['count'] = len(dataset_slices)
            new_dataset = DataSetEntity.parse_obj(dataset_dict)
            new_datasets.append(new_dataset)
        return new_datasets, pagination, 'Filter datasets succeed'

    def get_datasets_with_fuzzy(self, **kwargs) -> Tuple[List[DataSetEntity], str]:
        # userid = request_context.current_user.userid
        userid = 1
        name = kwargs.get('name')
        datasets = self.dataset_repository.get_datasets_with_fuzzy(userid, name)
        return datasets, 'Get datasets succeed'

    def update_dataset(self, **kwargs) -> Tuple[Optional[LabelEntity], str]:
        dataset_id = kwargs['dataset_id']
        dataset_data = kwargs['dataset_data']

        updated_count = self.dataset_repository.update(dataset_id, dataset_data)
        if not updated_count:
            return None, 'Update dataset failed'
        dataset = self.dataset_repository.get(dataset_id)
        return dataset, 'Update dataset succeed'

    def add_slices(self, **kwargs) -> Tuple[int, str]:
        dataset_id = kwargs['dataset_id']
        slice_ids = kwargs['slice_ids']
        dataset_slices = self.dataset_repository.get_dataset_slices_by_dataset(dataset_id)
        exist_slice_ids = [dataset_slice.slice_id for dataset_slice in dataset_slices]
        new_slice_ids = [slice_id for slice_id in slice_ids if slice_id not in exist_slice_ids]
        added_count = self.dataset_repository.add_slices(dataset_id, new_slice_ids)
        return added_count, 'Add slices to datasets succeed'

    def copy_dataset(self, dataset_id: int) -> Tuple[Optional[DataSetEntity], str]:
        old_dataset = self.dataset_repository.get(dataset_id)
        if not old_dataset:
            return None, 'No dataset selected'

        new_dataset = self.dataset_repository.save(DataSetEntity.parse_obj(old_dataset.dict(exclude={'id'})))

        old_dataset_slices = self.dataset_repository.get_dataset_slices_by_dataset(dataset_id)
        new_dataset_slices = []
        for old_dataset_slice in old_dataset_slices:
            new = old_dataset_slice.dict(exclude={'id'})
            new['dataset_id'] = new_dataset.id
            new_en = DataSetSliceEntity.parse_obj(new)
            new_dataset_slices.append(new_en)
        success = self.dataset_repository.batch_save_dataset_slice(new_dataset_slices)
        if not success:
            return None, 'Copy dataset Failed'
        return new_dataset, 'Copy dataset succeed'

    def delete_dataset(self, dataset_id: int) -> Tuple[int, str]:
        deleted_count = self.dataset_repository.delete(dataset_id)
        if not deleted_count:
            return deleted_count, 'No dataset deleted'
        return deleted_count, 'Delete dataset succeed'

    def create_filter_template(self, **kwargs) -> Tuple[Optional[FilterTemplateEntity], str]:
        filter_template_to_save = FilterTemplateEntity.parse_obj(kwargs)
        new_filter_template = self.filter_template_repository.save(filter_template_to_save)
        if not new_filter_template:
            return None, 'Create filter template failed'
        return new_filter_template, 'Create filter template succeed'

    def delete_filter_template(self, filter_template_id: int) -> Tuple[int, str]:
        deleted_count = self.filter_template_repository.delete(filter_template_id)
        if not deleted_count:
            return deleted_count, 'No filter template deleted'
        return deleted_count, 'Delete filter template succeed'

    def get_filter_template_by_id(self, filter_template_id: int) -> Tuple[Optional[FilterTemplateEntity], str]:
        filter_template = self.filter_template_repository.get(filter_template_id)
        if not filter_template:
            return None, 'No filter template'
        return filter_template, 'Get filter template succeed'

    def get_filter_templates(self) -> Tuple[List[FilterTemplateEntity], str]:
        filter_templates = self.filter_template_repository.gets(None)
        return filter_templates, 'Get filter templates succeed'

    def get_tile(self, **kwargs) -> Tuple[str, str]:
        slice_key = kwargs['slice_key']
        slice_name = kwargs['slice_name']
        x = kwargs['x']
        y = kwargs['y']
        z = kwargs['z']

        tile_dir = get_tile_dir(slice_key)
        os.makedirs(tile_dir, exist_ok=True)
        tile_path = get_tile_path(slice_key, x, y, z)

        if not fs.path_exists(tile_path):
            slice_path = get_slice_path(slice_key, slice_name)
            slide = open_slide(slice_path)
            tile_image = slide.get_tile(x, y, z)
            tile_image.save(tile_path)
        return tile_path, 'Get tile succeed'

    def get_analyses(self, **kwargs) -> Tuple[List[AnalysisVO], dict, str]:
        page = kwargs['page']
        per_page = kwargs['per_page']
        slice_id = kwargs['slice_id']
        userid = kwargs.get('userid')
        analyses, pagination = self.analysis_repository.get_analyses(page, per_page, slice_id, userid)
        analyses_hack = [AnalysisVO.parse_obj({**analysis.dict(), 'delete_permission': analysis.userid == 1}) for analysis in analyses]
        return analyses_hack, pagination, 'Get analyses success'

    def get_analysis(self, analysis_id: int) -> Tuple[Optional[AnalysisEntity], str]:
        analysis = self.analysis_repository.get(analysis_id)
        return analysis, 'get analysis success'

    def create_analysis(self, **kwargs) -> Tuple[Optional[AnalysisEntity], str]:
        analysis = self.analysis_repository.save(AnalysisEntity.parse_obj(kwargs))
        if analysis:
            return analysis, 'create analysis success'
        return None, 'create analysis failed'

    def delete_analysis(self, analysis_id: int) -> Tuple[int, str]:
        analysis = self.analysis_repository.get(analysis_id)

        deleted_count = self.analysis_repository.delete(analysis_id)
        if deleted_count:
            fs.remove_dir(get_db_dir(analysis.slice_key, analysis.key))
            return deleted_count, 'Deleted analysis succeed'
        return deleted_count, 'Deleted analysis failed'
