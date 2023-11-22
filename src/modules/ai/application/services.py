import logging
import time

from setting import RANK_AI_TASK
from src.modules.ai.application import tasks
from src.modules.ai.domain.services import AiDomainService
from src.seedwork.application.responses import AppResponse
from src.infra.cache import cache

logger = logging.getLogger(__name__)


class AiService(object):

    def __init__(self, domain_service: AiDomainService):
        self.domain_service = domain_service

    def start_ai_analysis(self, **kwargs) -> AppResponse:
        result = tasks.run_ai_task(**kwargs)
        rank = cache.get(RANK_AI_TASK, [])
        rank.append(task_id := result.task_id)
        cache.set(RANK_AI_TASK, rank)
        return AppResponse(message='ai start succeed', data={'task_id': task_id})

    def run_ai_task(self, **kwargs):
        start_time = time.time()
        logger.info(f'收到任务{kwargs["slice_id"]}')
        print(f'收到任务{kwargs["slice_id"]}')

        time.sleep(10000)

        # result = self.domain_service.run_tct(task)

        alg_time = time.time() - start_time
        logger.info(f'任务{kwargs["slice_id"]} - 算法部分完成,耗时{alg_time}')

        # self.domain_service.create_ai_marks(
        #     cell_marks=[mark.to_dict() for mark in result.cell_marks],
        #     roi_marks=[mark.to_dict() for mark in result.roi_marks],
        #     skip_mark_to_tile=task.ai_type in [AIType.bm]
        # )

        # self.domain_service.create_ai_record()

        total_time = time.time() - start_time
        logger.info(f'任务{kwargs["slice_id"]} - 全部完成,耗时{total_time}')

        return AppResponse(message='ai analysis succeed')

    def polling(self, task_id: str) -> AppResponse:
        err_msg, result = self.domain_service.get_ai_task_result(task_id)
        return AppResponse(err_code=1 if err_msg else 0, message=err_msg, data=result)
