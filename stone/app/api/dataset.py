from apiflask import APIBlueprint

from stone.app.base_schema import NameFuzzyQuery
from stone.app.schema.dataset import DataSetPageQuery, DataSetFilter, ListDataSetOut, SingleDataSetOut, DataSetIn, DataSetIdsOut, DataSetAndSliceIdsIn
from stone.app.service_factory import AppServiceFactory

dataset_blueprint = APIBlueprint('数据集', __name__, url_prefix='/datasets')


@dataset_blueprint.get('')
@dataset_blueprint.input(NameFuzzyQuery, location='query')
@dataset_blueprint.output(ListDataSetOut)
@dataset_blueprint.doc(summary='数据集模糊搜索', security='ApiAuth')
def get_datasets_with_fuzzy(query_data):
    res = AppServiceFactory.slice_service.get_datasets_with_fuzzy(**query_data)
    return res.response


@dataset_blueprint.post('/filter')
@dataset_blueprint.input(DataSetPageQuery, location='query')
@dataset_blueprint.input(DataSetFilter, location='json')
@dataset_blueprint.output(ListDataSetOut)
@dataset_blueprint.doc(summary='数据集列表', security='ApiAuth')
def filter_datasets(query_data, json_data):
    res = AppServiceFactory.slice_service.filter_datasets(**{'page_query': query_data, 'filter': json_data})
    return res.response


@dataset_blueprint.put('/add-slices')
@dataset_blueprint.input(DataSetAndSliceIdsIn, location='json')
@dataset_blueprint.output(DataSetIdsOut)
@dataset_blueprint.doc(summary='添加切片', security='ApiAuth')
def add_slices(json_data):
    res = AppServiceFactory.slice_service.add_slices(**json_data)
    return res.response


@dataset_blueprint.get('/<int:dataset_id>')
@dataset_blueprint.output(SingleDataSetOut)
@dataset_blueprint.doc(summary='数据集详情', security='ApiAuth')
def get_dataset(dataset_id):
    res = AppServiceFactory.slice_service.get_dataset(dataset_id)
    return res.response


@dataset_blueprint.put('/<int:dataset_id>')
@dataset_blueprint.input(DataSetIn, location='json')
@dataset_blueprint.output(SingleDataSetOut)
@dataset_blueprint.doc(summary='更新数据集', security='ApiAuth')
def update_dataset(dataset_id, json_data):
    res = AppServiceFactory.slice_service.update_dataset(**{'dataset_id': dataset_id, 'dataset_data': json_data})
    return res.response


@dataset_blueprint.delete('/<int:dataset_id>')
@dataset_blueprint.output(DataSetIdsOut)
@dataset_blueprint.doc(summary='删除数据集', security='ApiAuth')
def delete_dataset(dataset_id):
    res = AppServiceFactory.slice_service.delete_dataset(dataset_id)
    return res.response


@dataset_blueprint.post('/copy/<int:dataset_id>')
@dataset_blueprint.output(SingleDataSetOut)
@dataset_blueprint.doc(summary='复制数据集', security='ApiAuth')
def copy_dataset(dataset_id):
    res = AppServiceFactory.slice_service.copy_dataset(dataset_id)
    return res.response
