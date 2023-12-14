from apiflask import APIBlueprint
from stone.app.schema.ft import FilterTemplateIn, SingleFilterTemplateOut, ListFilterTemplateOut
from stone.app.service_factory import AppServiceFactory

filter_template_blueprint = APIBlueprint('筛选模板', __name__, url_prefix='/filter-templates')


@filter_template_blueprint.get('')
@filter_template_blueprint.output(ListFilterTemplateOut)
@filter_template_blueprint.doc(summary='筛选模板列表', security='ApiAuth')
def get_filter_templates():
    res = AppServiceFactory.slice_service.get_filter_templates()
    return res.response


@filter_template_blueprint.get('/<int:filter_template_id>')
@filter_template_blueprint.output(SingleFilterTemplateOut)
@filter_template_blueprint.doc(summary='筛选模板详情', security='ApiAuth')
def get_filter_template(filter_template_id):
    res = AppServiceFactory.slice_service.get_filter_template(filter_template_id)
    return res.response


@filter_template_blueprint.post('')
@filter_template_blueprint.input(FilterTemplateIn, location='json')
@filter_template_blueprint.output(SingleFilterTemplateOut)
@filter_template_blueprint.doc(summary='保存筛选模板', security='ApiAuth')
def create_filter_template(json_data):
    res = AppServiceFactory.slice_service.create_filter_template(**json_data)
    return res.response


@filter_template_blueprint.delete('/<int:filter_template_id>')
@filter_template_blueprint.output(SingleFilterTemplateOut)
@filter_template_blueprint.doc(summary='删除筛选模板', security='ApiAuth')
def delete_filter_template(filter_template_id):
    res = AppServiceFactory.slice_service.delete_filter_template(filter_template_id)
    return res.response
