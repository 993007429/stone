from apiflask import APIBlueprint

from stone.app.base_schema import NameFuzzyQuery, APIAffectedCountOut
from stone.app.schema.label import LabelPageQuery, LabelFilter, LabelIn, ApiSingleLabelOut, ApiListLabelOut
from stone.app.service_factory import AppServiceFactory

label_blueprint = APIBlueprint('标签', __name__, url_prefix='/labels')


@label_blueprint.get('')
@label_blueprint.input(NameFuzzyQuery, location='query')
@label_blueprint.output(ApiListLabelOut)
@label_blueprint.doc(summary='标签列表模糊筛选', security='ApiAuth')
def get_labels_with_fuzzy(query_data):
    res = AppServiceFactory.slice_service.get_labels_with_fuzzy(**query_data)
    return res.response


@label_blueprint.post('/filter')
@label_blueprint.input(LabelPageQuery, location='query')
@label_blueprint.input(LabelFilter, location='json')
@label_blueprint.output(ApiListLabelOut)
@label_blueprint.doc(summary='标签列表分页筛选', security='ApiAuth')
def filter_labels(query_data, json_data):
    res = AppServiceFactory.slice_service.filter_labels(**{'page_query': query_data, 'filter': json_data})
    return res.response


@label_blueprint.post('')
@label_blueprint.input(LabelIn, location='json')
@label_blueprint.output(ApiSingleLabelOut)
@label_blueprint.doc(summary='创建标签', security='ApiAuth')
def create_label(json_data):
    res = AppServiceFactory.slice_service.create_label(**json_data)
    return res.response


@label_blueprint.get('/<int:label_id>')
@label_blueprint.output(ApiSingleLabelOut)
@label_blueprint.doc(summary='标签详情', security='ApiAuth')
def get_label(label_id):
    res = AppServiceFactory.slice_service.get_label(label_id)
    return res.response


@label_blueprint.put('/<int:label_id>')
@label_blueprint.input(LabelIn, location='json')
@label_blueprint.output(ApiSingleLabelOut)
@label_blueprint.doc(summary='更新标签', security='ApiAuth')
def update_label(label_id, json_data):
    res = AppServiceFactory.slice_service.update_label(**{'label_id': label_id, 'label_data': json_data})
    return res.response


@label_blueprint.delete('/<int:label_id>')
@label_blueprint.output(APIAffectedCountOut)
@label_blueprint.doc(summary='删除标签', security='ApiAuth')
def delete_label(label_id):
    res = AppServiceFactory.slice_service.delete_label(label_id)
    return res.response
