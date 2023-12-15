import ast
import logging
import os
import time
import uuid

from setting import RANK_AI_TASK
from stone.app.request_context import request_context
from stone.infra.fs import fs
from stone.libs.heimdall.dispatch import open_slide
from stone.modules.ai.domain.enum import AnalysisStat, AIModel
from stone.modules.ai.domain.services import AiDomainService
from stone.modules.slice.application.services import SliceService
from stone.seedwork.application.responses import AppResponse
from stone.utils.get_path import get_slice_dir, get_roi_dir

logger = logging.getLogger(__name__)


class AiService(object):
    RANK_AI_TASK = RANK_AI_TASK

    def __init__(self, domain_service: AiDomainService, slice_service: SliceService):
        self.domain_service = domain_service
        self.slice_service = slice_service

    def start_ai_analysis(self, **kwargs) -> AppResponse[dict]:
        slice_id = kwargs['slice_id']
        ai_model = kwargs['ai_model']
        model_version = kwargs['model_version']

        result = self.run_ai_task(slice_id, ai_model, model_version)
        if result.err_code != 0:
            return AppResponse(err_code=result.err_code, message=result.message)
        task_id = 1

        # result = tasks.run_ai_task(slice_id, ai_model, model_version)
        # rank = cache.get(self.RANK_AI_TASK, [])
        # rank.append(task_id := result.task_id)
        # cache.set(self.RANK_AI_TASK, rank)
        return AppResponse(message='Ai start succeed', data={'task_id': task_id})

    def run_ai_task(self, slice_id: int, ai_model: str, model_version: str) -> AppResponse[dict]:
        res = self.slice_service.get_slice(slice_id)
        if res.err_code != 0:
            return res
        slice_ = res.data['slice']
        slice_key, slice_name = slice_['key'], slice_['name']
        slice_path = os.path.join(get_slice_dir(slice_key), slice_name)

        threshold = 1

        start_time = time.time()
        logger.info(f'收到任务 {slice_id}')

        # groups = self.domain_service.get_mark_groups(template_id=task_param.template_id)
        groups = []
        group_name_to_id = {group['label']: int(group['id']) for group in groups}

        model = self.domain_service.get_model(ai_model, model_version, threshold)
        if not model:
            return AppResponse(err_code=1, message=f'Model does not exist: {ai_model}_{model_version}')

        if ai_model in [AIModel.tct1, AIModel.tct2]:
            result = self.domain_service.run_tct(model, ai_model, slice_path)
        elif ai_model in [AIModel.lct1, AIModel.lct2]:
            result = self.domain_service.run_lct(model, ai_model, slice_path)
        elif ai_model == AIModel.tbs_dna:
            result = self.domain_service.run_tbs_dna(model, ai_model, slice_path)
        # elif ai_model == AIModel.dna_ploidy:
        #     result = self.domain_service.run_dna_ploidy(task_param)
        # elif ai_model == AIModel.her2:
        #     result = self.domain_service.run_her2(task_param, group_name_to_id)

        alg_time = time.time() - start_time
        logger.info(f'任务 {slice_id} - 算法部分完成,耗时{alg_time}')

        analysis_data = dict(
            key=uuid.uuid4().hex,
            ai_model=ai_model,
            model_version=model_version,
            status=AnalysisStat.success.value,
            time_consume=alg_time,
            userid=request_context.current_user.userid if request_context.current_user else 1,
            username=request_context.current_user.username if request_context.current_user else 'sa',
            slice_id=slice_id,
            slice_key=slice_key,
            ai_suggest=result.ai_suggest,
        )

        analysis_res = self.slice_service.create_analysis(**analysis_data)
        if analysis_res.err_code != 0:
            return analysis_res
        analysis = analysis_res.data['analysis']

        success = self.domain_service.create_ai_marks(
            analysis_key=analysis.key,
            slice_key=slice_key,
            ai_model=ai_model,
            model_version=model_version,
            cell_marks=[mark.dict() for mark in result.cell_marks],
            roi_marks=[mark.dict() for mark in result.roi_marks],
            skip_mark_to_tile=ai_model in [AIModel.bm]
        )

        if not success:
            return AppResponse(err_code=1, message='Ai analysis failed at creating marks')

        total_time = time.time() - start_time
        logger.info(f'任务 {slice_id} - 全部完成,耗时{total_time}')

        return AppResponse(message='Ai analysis succeed', data={'analysis_id': analysis.id})

    def get_task_status(self, task_id: str) -> AppResponse[dict]:
        message, task_status = self.domain_service.get_task_status(task_id)
        return AppResponse(message=message, data={'task_status': task_status})

    def get_roi(self, **kwargs) -> AppResponse[dict]:
        analysis_id = kwargs['analysis_id']
        roi_args = ast.literal_eval(kwargs['roi_args'])
        roi_id = kwargs['roi_id']

        analysis_res = self.slice_service.get_analysis(analysis_id)
        analysis = analysis_res.data['analysis']

        roi_dir = get_roi_dir(analysis.slice_key, analysis.key)
        os.makedirs(roi_dir, exist_ok=True)
        roi_path = os.path.join(roi_dir, f'{roi_id}.jpeg')

        if not fs.path_exists(roi_path):
            res = self.slice_service.get_slice(analysis.slice_id)
            if res.err_code != 0:
                return res
            slice_ = res.data['slice']
            slice_path = os.path.join(get_slice_dir(analysis.slice_key), slice_['name'])
            slide = open_slide(slice_path)
            tile_image = slide.get_roi(roi_args)
            tile_image.save(roi_path)

        return AppResponse(message='Get roi succeed', data={'roi_path': roi_path})

    def get_marks(self, analysis_id: int) -> AppResponse[dict]:
        marks, total, message = self.domain_service.get_marks(analysis_id)
        return AppResponse(message=message, data={'marks': marks, 'total': total})
