from apiflask import APIBlueprint, FileSchema
from flask import send_file

from stone.app.base_schema import APIAffectedCountOut
from stone.app.permission import permission_required
from stone.app.schema.analysis import AnalysesQuery, APIListAnalysesOut, QueryMarksIn, QueryRoiIn, \
    APIListMarkOut, APISingleAnalysesOut
from stone.app.service_factory import AppServiceFactory
from stone.modules.user.infrastructure.permissions import DeleteAnalysisPermission

analysis_blueprint = APIBlueprint('ai处理记录', __name__, url_prefix='/analyses')


@analysis_blueprint.get('')
@analysis_blueprint.input(AnalysesQuery, location='query')
@analysis_blueprint.output(APIListAnalysesOut)
@analysis_blueprint.doc(summary='AI处理记录列表', security='ApiAuth')
def get_analyses(query_data):
    res = AppServiceFactory.ai_service.get_analyses(**query_data)
    return res.response


@analysis_blueprint.get('/<int:analysis_id>')
@analysis_blueprint.output(APISingleAnalysesOut)
@analysis_blueprint.doc(summary='AI处理详情', security='ApiAuth')
def get_analysis(analysis_id):
    res = AppServiceFactory.ai_service.get_analysis(analysis_id)
    return res.response


@analysis_blueprint.delete('/<int:analysis_id>')
@permission_required([DeleteAnalysisPermission])
@analysis_blueprint.output(APIAffectedCountOut)
@analysis_blueprint.doc(summary='删除AI处理记录', security='ApiAuth')
def delete_analysis(analysis_id):
    res = AppServiceFactory.ai_service.delete_analysis(analysis_id)
    return res.response


@analysis_blueprint.get('/roi')
@analysis_blueprint.input(QueryRoiIn, location='query')
@analysis_blueprint.output(FileSchema(type='string', format='binary'), content_type='image/png')
@analysis_blueprint.doc(summary='ROI', security='ApiAuth')
def get_roi(query_data):
    res = AppServiceFactory.ai_service.get_roi(**query_data)
    return send_file(res.data.get('roi_path'))


@analysis_blueprint.get('/marks')
@analysis_blueprint.input(QueryMarksIn, location='query')
@analysis_blueprint.output(APIListMarkOut)
@analysis_blueprint.doc(summary='AI计算marks', security='ApiAuth')
def get_marks(query_data):
    res = AppServiceFactory.ai_service.get_marks(**query_data)
    return res.response
