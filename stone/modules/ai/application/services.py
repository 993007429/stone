import logging
import time

from setting import RANK_AI_TASK
from stone.app.request_context import request_context
from stone.modules.ai.domain.services import AiDomainService
from stone.modules.ai.domain.value_objects import TaskParam, AIType
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

        result.task_id = 1
        rank = cache.get(self.RANK_AI_TASK, [])
        rank.append(task_id := result.task_id)
        cache.set(self.RANK_AI_TASK, rank)
        return AppResponse(message='Ai start succeed', data={'task_id': task_id})

    def run_ai_task(self, task_param: TaskParam) -> AppResponse:
        start_time = time.time()
        logger.info(f'收到任务1 {task_param.slice_id}')

        # groups = self.domain_service.get_mark_groups(template_id=task_param.template_id)
        groups = []
        group_name_to_id = {group['label']: int(group['id']) for group in groups}

        if task_param.ai_model in [AIType.tct1, AIType.tct2]:
            result = self.domain_service.run_tct(task_param)
        elif task_param.ai_model in [AIType.lct1, AIType.lct2]:
            result = self.domain_service.run_lct(task_param)
        elif task_param.ai_model == AIType.dna:
            result = self.domain_service.run_tbs_dna(task_param)
        elif task_param.ai_model == AIType.dna_ploidy:
            result = self.domain_service.run_dna_ploidy(task_param)
        elif task_param.ai_model == AIType.her2:
            result = self.domain_service.run_her2(task_param, group_name_to_id)

        alg_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 算法部分完成,耗时{alg_time}')

        analysis = dict(
            userid=request_context.current_user.userid,
            username=request_context.current_user.username,
            slice_id=task_param.slice_id,
            ai_model=task_param.ai_model,
            model_version=task_param.model_version,
            status='',
            time_consume=alg_time
        )

        self.domain_service.create_analysis(**analysis)

        self.domain_service.create_ai_marks(
            ai_model=task_param.ai_model,
            slide_path=task_param.slide_path,
            cell_marks=[mark.dict() for mark in result.cell_marks],
            roi_marks=[mark.dict() for mark in result.roi_marks],
            skip_mark_to_tile=task_param.ai_model in [AIType.bm]
        )

        total_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 全部完成,耗时{total_time}')

        return AppResponse(message='Ai analysis succeed')

    def polling(self, task_id: str) -> AppResponse:
        err_msg, result = self.domain_service.get_ai_task_result(task_id)
        return AppResponse(err_code=1 if err_msg else 0, message=err_msg, data=result)

    def get_analyses(self, **kwargs) -> AppResponse:
        self.domain_service.get_analyses(**kwargs)
        return AppResponse()

    def get_analysis(self, analysis_id: int) -> AppResponse:
        self.domain_service.get_analysis(analysis_id)
        return AppResponse()
