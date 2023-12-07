from apiflask import Schema
from marshmallow.fields import Integer, String, DateTime, List, Nested, Bool, Dict, Float
from marshmallow.validate import OneOf

from stone.app.base_schema import DurationField, PageQuery, PaginationSchema, Coordinate
from stone.modules.ai.domain.enum import AnalysisStat


class AnalysesQuery(PageQuery):
    slice_id = Integer(required=True)
    userid = Integer(required=False)


class AnalysisOut(Schema):
    id = Integer(required=True)
    userid = Integer(required=True)
    username = String(required=True)
    slice_id = Integer(required=True)
    ai_model = String(required=True)
    model_version = String(required=True)
    status = String(required=True, validate=[OneOf([AnalysisStat.success.value, AnalysisStat.failed.value])])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    time_consume = DurationField(required=True)
    delete_permission = Bool(required=True)


class ListAnalysesOut(Schema):
    analyses = List(Nested(AnalysisOut()))


class APIListAnalysesOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListAnalysesOut())
    pagination = Nested(PaginationSchema())


class QueryRoiIn(Schema):
    analysis_id = Integer(required=True, description='AI分析记录ID')
    roi_args = String(required=True, description='Roi 参数')
    roi_id = Integer(required=True, description='Roi ID')


class QueryMarksIn(Schema):
    analysis_id = Integer(required=True, description='AI分析记录ID')


class AiResult(Schema):
    index = Integer(required=True)
    neg_tumor = Integer(required=True)
    normal_cell = Integer(required=True)
    pos_tumor = Integer(required=True)
    total = Integer(required=True)
    whole_slide = Integer(required=True)

id=1732284570046439424 created_at=datetime.datetime(2023, 12, 6, 11, 30, 41) last_modified=datetime.datetime(2023, 12, 6, 11, 30, 41) group=None is_in_manual=None position={'x': [], 'y': []} method='rectangle' is_export=1 remark=None ai_result={'cell_num': 5118, 'clarity': 1.0, 'slide_quality': 0, 'diagnosis': ['阴性', '-样本不满意'], 'microbe': [], 'cells': {'ASCUS': {'num': 14, 'data': [{'id': 0, 'path': {'x': [22916, 23304, 23304, 22916], 'y': [6004, 6004, 6392, 6392]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 3.4075772418873385e-05}, {'id': 1, 'path': {'x': [21523, 21911, 21911, 21523], 'y': [30664, 30664, 31052, 31052]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 3.3939249988179654e-05}, {'id': 2, 'path': {'x': [24603, 24991, 24991, 24603], 'y': [20494, 20494, 20882, 20882]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.6671605155570433e-05}, {'id': 3, 'path': {'x': [20144, 20532, 20532, 20144], 'y': [18169, 18169, 18557, 18557]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.6639481802703813e-05}, {'id': 4, 'path': {'x': [23235, 23623, 23623, 23235], 'y': [29693, 29693, 30081, 30081]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.545293864386622e-05}, {'id': 5, 'path': {'x': [22597, 22985, 22985, 22597], 'y': [8641, 8641, 9029, 9029]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.5184068363159895e-05}, {'id': 6, 'path': {'x': [31024, 31412, 31412, 31024], 'y': [28238, 28238, 28626, 28626]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.4914359528338537e-05}, {'id': 7, 'path': {'x': [23671, 24059, 24059, 23671], 'y': [11745, 11745, 12133, 12133]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.482596755726263e-05}, {'id': 8, 'path': {'x': [27587, 27975, 27975, 27587], 'y': [20830, 20830, 21218, 21218]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.466491423547268e-05}, {'id': 9, 'path': {'x': [21516, 21904, 21904, 21516], 'y': [27, 27, 415, 415]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.408158798061777e-05}, {'id': 10, 'path': {'x': [21350, 21738, 21738, 21350], 'y': [26104, 26104, 26492, 26492]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.3964203137438744e-05}, {'id': 11, 'path': {'x': [17622, 18010, 18010, 17622], 'y': [20394, 20394, 20782, 20782]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.2761611035093665e-05}, {'id': 12, 'path': {'x': [15349, 15737, 15737, 15349], 'y': [25632, 25632, 26020, 26020]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.1902978915022686e-05}, {'id': 13, 'path': {'x': [23762, 24150, 24150, 23762], 'y': [10450, 10450, 10838, 10838]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.1379237296059728e-05}]}, 'ASC-H': {'num': 4, 'data': [{'id': 14, 'path': {'x': [9389, 9777, 9777, 9389], 'y': [11551, 11551, 11939, 11939]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 0.0001048444100888446}, {'id': 15, 'path': {'x': [3354, 3742, 3742, 3354], 'y': [1940, 1940, 2328, 2328]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 4.812208135263063e-05}, {'id': 16, 'path': {'x': [9313, 9701, 9701, 9313], 'y': [7559, 7559, 7947, 7947]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.4239810954895802e-05}, {'id': 17, 'path': {'x': [12466, 12854, 12854, 12466], 'y': [26589, 26589, 26977, 26977]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.2448055460699834e-05}]}, 'LSIL': {'num': 0, 'data': []}, 'HSIL': {'num': 0, 'data': []}, 'AGC': {'num': 2, 'data': [{'id': 18, 'path': {'x': [12958, 13346, 13346, 12958], 'y': [30823, 30823, 31211, 31211]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.8997812478337437e-05}, {'id': 19, 'path': {'x': [6902, 7290, 7290, 6902], 'y': [5190, 5190, 5578, 5578]}, 'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0, 'cell_pos_prob': 2.658476478245575e-05}]}, '滴虫': {'num': 0, 'data': []}, '霉菌': {'num': 0, 'data': []}, '线索': {'num': 0, 'data': []}, '疱疹': {'num': 0, 'data': []}, '放线菌': {'num': 0, 'data': []}, '萎缩性改变': {'num': 0, 'data': []}, '修复细胞': {'num': 0, 'data': []}, '化生细胞': {'num': 0, 'data': []}, '腺上皮细胞': {'num': 0, 'data': []}, '炎性细胞': {'num': 0, 'data': []}}, 'whole_slide': 1} editable=None stroke_color='grey' fill_color=None mark_type=3 diagnosis=None radius=5.0 group_id=None area_id=None dashed=None doctor_diagnosis=None create_time=1701862241730.0
class TctAiResult(Schema):
    cell_num = Integer(required=True)
    clarity = Float(required=True)
    slide_quality = Integer(required=True)
    diagnosis = Integer(required=True)
    total = Integer(required=True)
    whole_slide = Integer(required=True)


class MarkOut(Schema):
    ai_result = Nested(TctAiResult(), description='算法结果')
    area_id = String(required=True, allow_none=True, description='算法区域标注id')
    create_time = String(required=True, allow_none=True, description='')
    dashed = String(required=True, allow_none=True, description='虚线为1，实线为0')
    diagnosis = String(required=True, allow_none=True, description='')
    doctor_diagnosis = String(required=True, allow_none=True, description='医生手工判读')
    editable = Integer(required=True, allow_none=True, description='可编辑为1，不可编辑为0')
    fill_color = String(required=True, allow_none=True, description='填充颜色')
    group_id = Integer(required=True, allow_none=True, description='标注组id')
    id = String(required=True, allow_none=True, description='')
    image = String(required=True, allow_none=True, description='')
    mark_type = Integer(required=True, allow_none=True, description='标注类型（手动标注1、算法标注2、算法标注区域3）')
    method = String(required=True, allow_none=True, description='AI分析记录ID')
    path = Nested(Coordinate(), description='AI分析记录ID')
    position = Nested(Coordinate(), description='AI分析记录ID')
    radius = Integer(required=True, allow_none=True, description='标注直径')
    remark = String(required=True, allow_none=True, description='AI分析记录ID')
    show_layer = Integer(required=True, allow_none=True, description='AI分析记录ID')
    stroke_color = String(required=True, allow_none=True, description='AI分析记录ID')


class ListMarkOut(Schema):
    marks = List(Nested(MarkOut()))


class APIListMarkOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListMarkOut())
