import logging
import os
import shutil
from functools import wraps
from inspect import getfullargspec
from typing import Optional, Type, Tuple, List

from celery.result import AsyncResult

from setting import RANK_AI_TASK
from src.app.request_context import request_context
from src.infra.cache import cache
from src.libs.heimdall.dispatch import open_slide
from src.celery.app import app as celery_app
from celery.exceptions import TimeoutError as CeleryTimeoutError
from src.modules.ai.domain.value_objects import Mark, ALGResult, TaskParam, AIType
from src.modules.ai.infrastructure.repositories import SQLAlchemyAiRepository
from src.seedwork.application.responses import AppResponse

logger = logging.getLogger(__name__)


def connect_slice_db():
    def deco(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            # _self: SliceAnalysisService = args[0]

            db_doc_path = kwargs['slide_path']
            db_template_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'slice.db')
            if not os.path.exists(db_doc_path):
                shutil.copyfile(db_template_path, db_doc_path)

            request_context.connect_slice_db(db_doc_path)

            # ai_type = request_context.ai_type
            #
            # if ai_type:
            #     template_id = 0
            #     if ai_type == AIType.label:
            #         slice_info = _self.slice_service.get_slice_info(case_id=case_id, file_id=file_id).data
            #         template_id = slice_info['templateId'] if slice_info else 0
            #         func_args = getfullargspec(f)[0]
            #         if 'template_id' in func_args and not kwargs.get('template_id'):
            #             kwargs['template_id'] = template_id

                # mark_table_suffix = _self.domain_service.get_mark_table_suffix(ai_type=ai_type, template_id=template_id)
                # _self.domain_service.repository.mark_table_suffix = mark_table_suffix
                # _self.domain_service.repository.create_mark_tables(ai_type=ai_type)

            # manual_table_suffix = AI_TYPE_MANUAL_MARK_TABLE_MAPPING.get(ai_type, 'human')
            # _self.domain_service.repository.manual.mark_table_suffix = manual_table_suffix
            # _self.domain_service.repository.manual.create_mark_tables(ai_type=ai_type)

            r = f(*args, **kwargs)

            request_context.close_slice_db()

            return r

        return wrapped

    return deco


class AiDomainService(object):
    RANK_AI_TASK = RANK_AI_TASK

    def __init__(self, repository: SQLAlchemyAiRepository):
        self.repository = repository

    def get_ai_task_result(self, task_id: str) -> Tuple[str, Optional[dict]]:
        try:
            result = AsyncResult(task_id, app=celery_app)
            task_queue = cache.get(self.RANK_AI_TASK, [])
            if result.ready():
                task_queue.remove(task_id)
                cache.set(self.RANK_AI_TASK, task_queue)
                return 'AI处理完成', {'done': True, 'rank': -1}

            rank = task_queue.index(task_id)
            return 'Ai处理在排队中', {'done': False, 'rank': rank}

        except CeleryTimeoutError:
            pass
        except Exception as e:
            logger.exception(e)
            return 'AI处理发生异常', {'done': True, 'rank': -1}

    def _select_model(self, ai_model, model_version) -> Optional[Type]:
        from src.modules.ai.libs.algorithms.TCTAnalysis_v2_2.tct_alg import (
            LCT_mobile_micro0324, LCT40k_convnext_nofz, LCT40k_convnext_HDX, LCT_mix80k0417_8)
        from src.modules.ai.libs.algorithms.TCTAnalysis_v3_1.tct_alg import TCT_ALG2

        return TCT_ALG2

    def run_tct(self, task_param: TaskParam):
        ai_model = task_param.ai_model
        model_version = task_param.model_version
        slide_path = task_param.slide_path

        rois = []
        # rois = task_param.rois
        # model_info = task_param.rois
        # threshold = model_info.get('ai_threshold')
        # model_type = model_info.get('model_type')
        # model_name = model_info.get('model_name')

        alg_class = self._select_model(ai_model, model_version)

        slide = open_slide(slide_path)

        roi_marks = []
        prob_dict = None
        for idx, roi in enumerate(rois or [task_param.new_default_roi()]):
            if alg_class.__name__ == 'TCT_ALG2':
                # config_path = ai_model + model_type if model_type.isdigit() else model_type
                config_path = ''
                threshold = 1
                alg_obj = alg_class(config_path=config_path, threshold=threshold)
                result = alg_obj.cal_tct(slide)

                from src.modules.ai.utils.tct import generate_ai_result2
                ai_result = generate_ai_result2(result=result, roiid=roi['id'])

            else:
                threshold = 1
                alg_obj = alg_class(threshold=threshold)
                result = alg_obj.cal_tct(slide)
                from src.modules.ai.utils.tct import generate_ai_result
                ai_result = generate_ai_result(result=result, roiid=roi['id'])

            from src.modules.ai.utils.prob import save_prob_to_file
            prob_dict = save_prob_to_file(slide_path=slide_path, result=result)

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

    @connect_slice_db()
    def create_ai_marks(
            self,
            slide_path: str,
            cell_marks: List[dict],
            roi_marks: List[dict],
            skip_mark_to_tile: bool = False
    ):
        pass









































