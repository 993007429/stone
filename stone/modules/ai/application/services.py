import logging
import time

from setting import RANK_AI_TASK
from stone.app.request_context import request_context
from stone.modules.ai.domain.enum import AnalysisStat, AIModel
from stone.modules.ai.domain.services import AiDomainService
from stone.modules.ai.domain.value_objects import TaskParam, ALGResult, Mark
from stone.modules.slice.application.services import SliceService
from stone.seedwork.application.responses import AppResponse
from stone.infra.cache import cache

logger = logging.getLogger(__name__)


class AiService(object):
    RANK_AI_TASK = RANK_AI_TASK

    def __init__(self, domain_service: AiDomainService, slice_service: SliceService):
        self.domain_service = domain_service
        self.slice_service = slice_service

    def start_ai_analysis(self, **kwargs) -> AppResponse:
        task_param = TaskParam(**kwargs)
        task_param.slide_path = 'D:\\data\\789.svs'
        # task_param.slide_path = self.slice_service.get_slice_path(task_param.slice_id).data
        # result = tasks.run_ai_task(task_param)
        result = self.run_ai_task(task_param)

        task_id = 1
        # rank = cache.get(self.RANK_AI_TASK, [])
        # rank.append(task_id := result.task_id)
        # cache.set(self.RANK_AI_TASK, rank)
        return AppResponse(message='Ai start succeed', data={'task_id': task_id})

    def run_ai_task(self, task_param: TaskParam) -> AppResponse:
        start_time = time.time()
        logger.info(f'收到任务1 {task_param.slice_id}')

        # groups = self.domain_service.get_mark_groups(template_id=task_param.template_id)
        groups = []
        group_name_to_id = {group['label']: int(group['id']) for group in groups}

        # if task_param.ai_model in [AIModel.tct1, AIModel.tct2]:
        #     result = self.domain_service.run_tct(task_param)
        # elif task_param.ai_model in [AIModel.lct1, AIModel.lct2]:
        #     result = self.domain_service.run_lct(task_param)
        # elif task_param.ai_model == AIModel.dna:
        #     result = self.domain_service.run_tbs_dna(task_param)
        # elif task_param.ai_model == AIModel.dna_ploidy:
        #     result = self.domain_service.run_dna_ploidy(task_param)
        # elif task_param.ai_model == AIModel.her2:
        #     result = self.domain_service.run_her2(task_param, group_name_to_id)

        alg_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 算法部分完成,耗时{alg_time}')

        # analysis_data = dict(
        #     userid=request_context.current_user.userid if request_context.current_user else 1,
        #     username=request_context.current_user.username if request_context.current_user else 'sa',
        #     slice_id=task_param.slice_id,
        #     ai_model=task_param.ai_model,
        #     model_version=task_param.model_version,
        #     status=AnalysisStat.success.value,
        #     time_consume=alg_time
        # )
        analysis_data = {'userid': 1, 'username': 'sa', 'slice_id': 0, 'ai_model': 'tct1', 'model_version': 'v2',
                    'status': 1, 'time_consume': 49.704445362091064}
        result = ALGResult(ai_suggest='阴性 -样本不满意 ', cell_marks=[], roi_marks=[
            Mark(id=1732284570046439424, position={'x': [], 'y': []},
                 ai_result={'cell_num': 5118, 'clarity': 1.0, 'slide_quality': 0, 'diagnosis': ['阴性', '-样本不满意'],
                            'microbe': [], 'cells': {'ASCUS': {'num': 14, 'data': [
                         {'id': 0, 'path': {'x': [22916, 23304, 23304, 22916], 'y': [6004, 6004, 6392, 6392]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 3.4075772418873385e-05},
                         {'id': 1, 'path': {'x': [21523, 21911, 21911, 21523], 'y': [30664, 30664, 31052, 31052]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 3.3939249988179654e-05},
                         {'id': 2, 'path': {'x': [24603, 24991, 24991, 24603], 'y': [20494, 20494, 20882, 20882]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.6671605155570433e-05},
                         {'id': 3, 'path': {'x': [20144, 20532, 20532, 20144], 'y': [18169, 18169, 18557, 18557]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.6639481802703813e-05},
                         {'id': 4, 'path': {'x': [23235, 23623, 23623, 23235], 'y': [29693, 29693, 30081, 30081]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.545293864386622e-05},
                         {'id': 5, 'path': {'x': [22597, 22985, 22985, 22597], 'y': [8641, 8641, 9029, 9029]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.5184068363159895e-05},
                         {'id': 6, 'path': {'x': [31024, 31412, 31412, 31024], 'y': [28238, 28238, 28626, 28626]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.4914359528338537e-05},
                         {'id': 7, 'path': {'x': [23671, 24059, 24059, 23671], 'y': [11745, 11745, 12133, 12133]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.482596755726263e-05},
                         {'id': 8, 'path': {'x': [27587, 27975, 27975, 27587], 'y': [20830, 20830, 21218, 21218]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.466491423547268e-05},
                         {'id': 9, 'path': {'x': [21516, 21904, 21904, 21516], 'y': [27, 27, 415, 415]}, 'image': 0,
                          'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424,
                          'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.408158798061777e-05},
                         {'id': 10, 'path': {'x': [21350, 21738, 21738, 21350], 'y': [26104, 26104, 26492, 26492]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.3964203137438744e-05},
                         {'id': 11, 'path': {'x': [17622, 18010, 18010, 17622], 'y': [20394, 20394, 20782, 20782]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.2761611035093665e-05},
                         {'id': 12, 'path': {'x': [15349, 15737, 15737, 15349], 'y': [25632, 25632, 26020, 26020]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.1902978915022686e-05},
                         {'id': 13, 'path': {'x': [23762, 24150, 24150, 23762], 'y': [10450, 10450, 10838, 10838]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.1379237296059728e-05}]}, 'ASC-H': {'num': 4, 'data': [
                         {'id': 14, 'path': {'x': [9389, 9777, 9777, 9389], 'y': [11551, 11551, 11939, 11939]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 0.0001048444100888446},
                         {'id': 15, 'path': {'x': [3354, 3742, 3742, 3354], 'y': [1940, 1940, 2328, 2328]}, 'image': 0,
                          'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424,
                          'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 4.812208135263063e-05},
                         {'id': 16, 'path': {'x': [9313, 9701, 9701, 9313], 'y': [7559, 7559, 7947, 7947]}, 'image': 0,
                          'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2, 'area_id': 1732284570046439424,
                          'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.4239810954895802e-05},
                         {'id': 17, 'path': {'x': [12466, 12854, 12854, 12466], 'y': [26589, 26589, 26977, 26977]},
                          'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                          'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                          'cell_pos_prob': 2.2448055460699834e-05}]}, 'LSIL': {'num': 0, 'data': []},
                                                     'HSIL': {'num': 0, 'data': []}, 'AGC': {'num': 2, 'data': [
                             {'id': 18, 'path': {'x': [12958, 13346, 13346, 12958], 'y': [30823, 30823, 31211, 31211]},
                              'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                              'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                              'cell_pos_prob': 2.8997812478337437e-05},
                             {'id': 19, 'path': {'x': [6902, 7290, 7290, 6902], 'y': [5190, 5190, 5578, 5578]},
                              'image': 0, 'editable': 0, 'dashed': 0, 'fillColor': '', 'mark_type': 2,
                              'area_id': 1732284570046439424, 'method': 'rectangle', 'strokeColor': 'red', 'radius': 0,
                              'cell_pos_prob': 2.658476478245575e-05}]}, '滴虫': {'num': 0, 'data': []},
                                                     '霉菌': {'num': 0, 'data': []}, '线索': {'num': 0, 'data': []},
                                                     '疱疹': {'num': 0, 'data': []}, '放线菌': {'num': 0, 'data': []},
                                                     '萎缩性改变': {'num': 0, 'data': []},
                                                     '修复细胞': {'num': 0, 'data': []},
                                                     '化生细胞': {'num': 0, 'data': []},
                                                     '腺上皮细胞': {'num': 0, 'data': []},
                                                     '炎性细胞': {'num': 0, 'data': []}}, 'whole_slide': 1},
                 fill_color=None, stroke_color='grey', mark_type=3, diagnosis=None, radius=5.0, area_id=None,
                 editable=None, group_id=None, method='rectangle', is_export=1)], slide_quality=0, cell_num=5118,
                           prob_dict={'NILM': 0.99216, 'ASC-US': 0.00709, 'LSIL': 0.00026, 'ASC-H': 0.00019,
                                      'HSIL': 1e-05, 'AGC': 0.0003}, err_msg=None)

        analysis, _ = self.domain_service.create_analysis(**analysis_data)
        if not analysis or not analysis.id:
            return AppResponse(message='Ai analysis failed at creating analysis')

        success = self.domain_service.create_ai_marks(
            analysis_id=analysis.id,
            ai_model=task_param.ai_model,
            model_version=task_param.model_version,
            slide_path=task_param.slide_path,
            cell_marks=[mark.dict() for mark in result.cell_marks],
            roi_marks=[mark.dict() for mark in result.roi_marks],
            skip_mark_to_tile=task_param.ai_model in [AIModel.bm]
        )

        if not success:
            return AppResponse(message='Ai analysis failed at creating marks')

        total_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 全部完成,耗时{total_time}')

        return AppResponse(message='Ai analysis succeed')

    def polling(self, task_id: str) -> AppResponse:
        err_msg, result = self.domain_service.get_ai_task_result(task_id)
        return AppResponse(err_code=1 if err_msg else 0, message=err_msg, data=result)

    def get_analyses(self, **kwargs) -> AppResponse[dict]:
        analyses, pagination, message = self.domain_service.get_analyses(**kwargs)
        return AppResponse(message=message, data={'analyses': [analysis.dict() for analysis in analyses]},  pagination=pagination)

    def get_analysis(self, analysis_id: int) -> AppResponse:
        self.domain_service.get_analysis(analysis_id)
        return AppResponse()
