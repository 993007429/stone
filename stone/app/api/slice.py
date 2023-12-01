from apiflask import APIBlueprint, FileSchema
from flask import send_from_directory

from stone.app.schema.slice import ListSliceOut, SlicePageQuery, SingleSliceOut, SliceFilter, SliceIdsOut, SliceIdsIn, \
    WSIIn, SliceId, ROIIn, ComparisonSliceFilter, ComparisonListSliceOut, SliceIn, SliceUploadIn, \
    SingleSliceUploadOut, SliceUpdateIn, SingleSliceFieldOut, SliceAndLabelIdsIn
from stone.app.service_factory import AppServiceFactory

slice_blueprint = APIBlueprint('切片', __name__, url_prefix='/slices')


@slice_blueprint.post('/filter')
@slice_blueprint.input(SlicePageQuery, location='query')
@slice_blueprint.input(SliceFilter, location='json')
@slice_blueprint.output(ListSliceOut)
@slice_blueprint.doc(summary='切片列表分页筛选', security='ApiAuth')
def filter_slices(query_data, json_data):
    res = AppServiceFactory.slice_service.filter_slices(**{'page_query': query_data, 'filter': json_data})
    return res.response


@slice_blueprint.post('/upload')
@slice_blueprint.input(SliceUploadIn, location='form_and_files')
@slice_blueprint.output(SingleSliceUploadOut)
@slice_blueprint.doc(summary='上传切片文件', security='ApiAuth')
def upload_slice(form_and_files_data):
    res = AppServiceFactory.slice_service.upload_slice(**form_and_files_data)
    return res.response


@slice_blueprint.post('')
@slice_blueprint.input(SliceIn, location='json')
@slice_blueprint.output(SingleSliceOut)
@slice_blueprint.doc(summary='创建切片', security='ApiAuth')
def create_slice(json_data):
    res = AppServiceFactory.slice_service.create_slice(**json_data)
    return res.response


@slice_blueprint.get('/<int:slice_id>')
@slice_blueprint.output(SingleSliceOut)
@slice_blueprint.doc(summary='切片详情', security='ApiAuth')
def get_slice(slice_id):
    res = AppServiceFactory.slice_service.get_slice(slice_id)
    return res.response


@slice_blueprint.delete('')
@slice_blueprint.input(SliceIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='批量删除切片', security='ApiAuth')
def delete_slices(json_data):
    res = AppServiceFactory.slice_service.delete_slices(**json_data)
    return res.response


@slice_blueprint.put('')
@slice_blueprint.input(SliceUpdateIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='批量更新切片', security='ApiAuth')
def update_slices(json_data):
    res = AppServiceFactory.slice_service.update_slices(**json_data)
    return res.response


@slice_blueprint.get('/comparison')
@slice_blueprint.input(SlicePageQuery, location='query')
@slice_blueprint.input(ComparisonSliceFilter, location='json')
@slice_blueprint.output(ComparisonListSliceOut)
@slice_blueprint.doc(summary='对比模式', security='ApiAuth')
def get_comparison_slices(query_data, json_data):
    res = AppServiceFactory.slice_service.get_comparison_slices(**query_data, **json_data)
    return res.response


@slice_blueprint.put('/add-labels')
@slice_blueprint.input(SliceAndLabelIdsIn, location='json')
@slice_blueprint.output(SliceIdsOut)
@slice_blueprint.doc(summary='添加标签', security='ApiAuth')
def add_labels(json_data):
    res = AppServiceFactory.slice_service.add_labels(**json_data)
    return res.response


@slice_blueprint.get('/WSI')
@slice_blueprint.input(WSIIn, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='切片全场图', security='ApiAuth')
def get_wsi(query_data):
    AppServiceFactory.slice_service.get_wsi(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')


@slice_blueprint.get('/ROI')
@slice_blueprint.input(ROIIn, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='ROI', security='ApiAuth')
def get_roi(query_data):
    AppServiceFactory.slice_service.get_roi(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')


@slice_blueprint.get('/label')
@slice_blueprint.input(SliceId, location='query')
@slice_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@slice_blueprint.doc(summary='切片标签', security='ApiAuth')
def get_label(query_data):
    AppServiceFactory.slice_service.get_label(**query_data)
    return send_from_directory('IMAGE_FOLDER', 'filename')


@slice_blueprint.post('/fields')
@slice_blueprint.output(SingleSliceFieldOut)
@slice_blueprint.doc(summary='切片字段', security='ApiAuth')
def get_slice_fields():
    res = AppServiceFactory.slice_service.get_slice_fields()
    return res.response