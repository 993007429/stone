import logging
from typing import Optional, Type, Tuple

from celery.bin.control import inspect
from celery.result import AsyncResult

from src.libs.heimdall.dispatch import open_slide
from src.celery.app import app as celery_app
from celery.exceptions import TimeoutError as CeleryTimeoutError
from src.modules.ai.domain.value_objects import Mark, ALGResult
from src.modules.ai.infrastructure.repositories import SQLAlchemyAiRepository

logger = logging.getLogger(__name__)


class AiDomainService(object):

    def __init__(self, repository: SQLAlchemyAiRepository):
        self.repository = repository

    def get_ai_task_result(self, task_id: str) -> Tuple[str, Optional[dict]]:
        try:
            result = AsyncResult(task_id, app=celery_app)
            if result.ready():
                return 'AI处理完成', {'done': True, 'rank': -1}
            # else:
            #     i = inspect(app=celery_app)
            #     tasks = i.reserved()
            #     pending_tasks = 0
            #     if tasks:
            #         pending_tasks = len(tasks[None])
            #     return 'Ai处理在排队中', {'done': False, 'rank': pending_tasks}

        except CeleryTimeoutError:
            pass
        except Exception as e:
            logger.exception(e)
            return 'AI处理发生异常', {'done': True, 'rank': -1}

    def _select_model(self, ai_model, model_version) -> Optional[Type]:
        pass

    def run_tct(self):
        slice_id = task.slice_id
        ai_model = task.ai_model
        model_version = task.model_version

        model_info = task.model_info
        threshold = model_info.get('ai_threshold')
        model_type = model_info.get('model_type')
        model_name = model_info.get('model_name')

        alg_class = self._select_model(ai_model, model_version)

        slide = open_slide(task.slide_path)

        roi_marks = []
        prob_dict = None
        for idx, roi in enumerate(task.rois or [task.new_default_roi()]):
            if alg_class.__name__ == 'TCT_ALG2':
                config_path = task.ai_type.ai_name + model_type if model_type.isdigit() else model_type
                alg_obj = alg_class(config_path=config_path, threshold=threshold)
                result = alg_obj.cal_tct(slide)

                from src.modules.ai.utils.tct import generate_ai_result2
                ai_result = generate_ai_result2(result=result, roiid=roi['id'])

            else:
                alg_obj = alg_class(threshold=threshold)
                result = alg_obj.cal_tct(slide)
                from src.modules.ai.utils.tct import generate_ai_result
                ai_result = generate_ai_result(result=result, roiid=roi['id'])

            from src.modules.ai.utils.prob import save_prob_to_file
            prob_dict = save_prob_to_file(slide_path=task.slide_path, result=result)

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                method='rectangle',
                mark_type=3,
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result
            ))

            ai_result = roi_marks[0].ai_result

            return ALGResult(
                roi_marks=roi_marks,
                ai_suggest=' '.join(ai_result['diagnosis']) + ' ' + ','.join(ai_result['microbe']),
                slide_quality=ai_result['slide_quality'],
                cell_num=ai_result['cell_num'],
                prob_dict=prob_dict
            )

































