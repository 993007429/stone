import logging
import time

from setting import RANK_AI_TASK
from src.modules.ai.application import tasks
from src.modules.ai.domain.services import AiDomainService
from src.modules.ai.domain.value_objects import TaskParam, AIType
from src.modules.slice.application.services import SliceService
from src.seedwork.application.responses import AppResponse
from src.infra.cache import cache

logger = logging.getLogger(__name__)


class AiService(object):
    RANK_AI_TASK = RANK_AI_TASK

    def __init__(self, domain_service: AiDomainService, slice_service: SliceService):
        self.domain_service = domain_service
        self.slice_service = slice_service

    def start_ai_analysis(self, **kwargs) -> AppResponse:
        task_param = TaskParam(**kwargs)
        task_param.slide_path = 'D:\\data\\123.sdpc'
        # task_param.slide_path = self.slice_service.get_slice_path(task_param.slice_id).data
        # result = tasks.run_ai_task(task_param)
        result = self.run_ai_task(task_param)
        rank = cache.get(self.RANK_AI_TASK, [])
        rank.append(task_id := result.task_id)
        cache.set(self.RANK_AI_TASK, rank)
        return AppResponse(message='ai start succeed', data={'task_id': task_id})

    def run_ai_task(self, task_param: TaskParam):
        start_time = time.time()
        logger.info(f'收到任务1 {task_param.slice_id}')

        if task_param.ai_model == AIType.tct:
            result = self.domain_service.run_tct(task_param)

        alg_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 算法部分完成,耗时{alg_time}')

        # self.domain_service.create_ai_marks(
        #     slide_path=task_param.slide_path,
        #     cell_marks=[mark.dict() for mark in result.cell_marks],
        #     roi_marks=[mark.dict() for mark in result.roi_marks],
        #     skip_mark_to_tile=task_param.ai_model in [AIType.bm]
        # )

        # self.domain_service.create_ai_record()

        total_time = time.time() - start_time
        logger.info(f'任务 {task_param.slice_id} - 全部完成,耗时{total_time}')

        return AppResponse(message='ai analysis succeed')

    def polling(self, task_id: str) -> AppResponse:
        err_msg, result = self.domain_service.get_ai_task_result(task_id)
        return AppResponse(err_code=1 if err_msg else 0, message=err_msg, data=result)













