import ast
import logging
import os
import shutil
import time
from functools import wraps
from typing import Optional, Tuple, List, Any

from celery.result import AsyncResult

import setting
from setting import RANK_AI_TASK
from stone.app.permission import permission_required
from stone.app.request_context import request_context
from stone.consts.her2 import Her2Consts
from stone.infra.cache import cache
from stone.infra.fs import fs
from stone.modules.ai.domain.enum import AIModel
from stone.modules.ai.libs.algorithms.DNA1.dna_alg import DNA_1020
from stone.libs.heimdall.dispatch import open_slide
from stone.celery.app import app as celery_app
from celery.exceptions import TimeoutError as CeleryTimeoutError

from stone.modules.ai.domain.consts import AI_TYPE_MANUAL_MARK_TABLE_MAPPING
from stone.modules.ai.domain.entities import MarkEntity, AnalysisEntity, AnalysisVO
from stone.modules.ai.domain.value_objects import Mark, ALGResult
from stone.modules.ai.infrastructure.repositories import SQLAlchemyAIRepository
from stone.modules.ai.utils.tct import generate_ai_result, generate_dna_ai_result
from stone.modules.user.infrastructure.permissions import DeleteAnalysisPermission
from stone.utils.get_path import get_slice_dir, get_db_path, get_db_dir, get_roi_dir
from stone.utils.id_worker import IdWorker

from stone.utils.load_yaml import load_yaml

logger = logging.getLogger(__name__)


def create_slice_db():
    def deco(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            _self: AiDomainService = args[0]

            ai_model = kwargs['ai_model']
            model_version = kwargs['model_version']
            slice_key = kwargs['slice_key']
            analysis_key = kwargs['analysis_key']
            db_path = get_db_path(slice_key, analysis_key, ai_model, model_version)
            db_template_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'slice.db')

            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            shutil.copyfile(db_template_path, db_path)

            request_context.connect_slice_db(db_path)

            #     template_id = 0
            #     if ai_type == AIModel.label:
            #         slice_info = _self.slice_service.get_slice_info(case_id=case_id, file_id=file_id).data
            #         template_id = slice_info['templateId'] if slice_info else 0
            #         func_args = getfullargspec(f)[0]
            #         if 'template_id' in func_args and not kwargs.get('template_id'):
            #             kwargs['template_id'] = template_id

            # mark_table_suffix = _self.repository.get_mark_table_suffix(ai_type=ai_type, template_id=template_id)

            _self.repository.mark_table_suffix = f'{ai_model}_{model_version}'
            _self.repository.create_mark_tables(ai_model)

            r = f(*args, **kwargs)

            request_context.close_slice_db()

            return r

        return wrapped

    return deco


def connect_slice_db():
    def deco(f):

        @wraps(f)
        def wrapped(*args, **kwargs):
            _self: AiDomainService = args[0]
            analysis_id = args[1]
            analysis = _self.repository.get_analysis_by_pk(analysis_id)
            ai_model = analysis.ai_model
            model_version = analysis.model_version
            analysis_key = analysis.key
            slice_key = analysis.slice_key

            _self.repository.mark_table_suffix = f'{analysis.ai_model}_{analysis.model_version}'

            db_path = get_db_path(slice_key, analysis_key, ai_model, model_version)

            request_context.connect_slice_db(db_path)

            r = f(*args, **kwargs)

            request_context.close_slice_db()

            return r

        return wrapped

    return deco


class AiDomainService(object):
    RANK_AI_TASK = RANK_AI_TASK

    def __init__(self, repository: SQLAlchemyAIRepository):
        self.repository = repository

    def get_ai_task_result(self, task_id: str) -> Tuple[str, Optional[dict]]:
        try:
            result = AsyncResult(task_id, app=celery_app)
            task_queue = cache.get(self.RANK_AI_TASK, [])
            if result.ready():
                task_queue.remove(task_id)
                cache.set(self.RANK_AI_TASK, task_queue)
                analysis_id = result.result.data['analysis_id']
                return 'AI analysis completed', {'done': True, 'rank': -1, 'analysis_id': analysis_id}

            rank = task_queue.index(task_id)
            return 'AI analysis in queue', {'done': False, 'rank': rank}

        except CeleryTimeoutError:
            pass
        except ValueError:
            return 'AI analysis has been completed', {'done': True, 'rank': -2}
        except Exception as e:
            logger.exception(e)
            return 'An exception occurred in AI analysis', {'done': True, 'rank': -3}

    def new_default_roi(self) -> dict:
        return {
            'id': IdWorker.new_mark_id_worker().get_new_id(),
            'x': [],
            'y': []
        }

    def get_model(self, ai_model: str, model_version: str, threshold: float) -> Any:
        from stone.modules.ai.libs.algorithms.TCTAnalysis_v2_2.tct_alg import AlgBase
        from stone.modules.ai.libs.algorithms.TCTAnalysis_v3_1.tct_alg import TCT_ALG2

        yams_path = os.path.join(setting.PROJECT_DIR, 'yams')
        deploy_yaml = load_yaml(os.path.join(yams_path, 'deploy.yaml'))
        try:
            yaml_file = deploy_yaml[ai_model][model_version]
        except KeyError:
            return None

        if ai_model == AIModel.tct1:
            config_path = os.path.join(yams_path, 'tct1', yaml_file)
            return AlgBase(config_path=config_path, threshold=threshold)
        elif ai_model == AIModel.tct2:
            config_path = os.path.join(yams_path, 'tct2', yaml_file)
            return TCT_ALG2(config_path=config_path, threshold=threshold)

    def run_tct(self, alg_model: Any, ai_model: str, slice_path: str) -> Optional[ALGResult]:
        rois = []

        slide = open_slide(slice_path)

        roi_marks = []
        prob_dict = None
        for idx, roi in enumerate(rois or [self.new_default_roi()]):
            result = alg_model.cal_tct(slide)

            if ai_model == AIModel.tct2:
                from stone.modules.ai.utils.tct import generate_ai_result2
                ai_result = generate_ai_result2(result=result, roiid=roi['id'])
            else:
                from stone.modules.ai.utils.tct import generate_ai_result
                ai_result = generate_ai_result(result=result, roiid=roi['id'])

            from stone.modules.ai.utils.prob import save_prob_to_file
            prob_dict = save_prob_to_file(slice_path=slice_path, result=result)

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

    def run_lct(self, alg_model: Any, ai_model: str, slice_path: str) -> ALGResult:
        return self.run_tct(alg_model, ai_model, slice_path)

    def run_tbs_dna(self) -> ALGResult:
        ai_model = task_param.ai_model
        model_version = task_param.model_version
        slice_path = task_param.slice_path

        dna_alg_model = DNA_1020()

        rois = []
        threshold = 1

        alg_model = self.get_model(ai_model, model_version, threshold)

        slide = open_slide(slice_path)

        roi_marks = []
        prob_dict = None

        for idx, roi in enumerate(rois or [task_param.new_default_roi()]):
            tbs_result = alg_model.cal_tct(slide)
            dna_result = dna_alg_model.dna_test(slide)

            from stone.modules.ai.utils.prob import save_prob_to_file
            prob_dict = save_prob_to_file(slice_path=task_param.slice_path, result=tbs_result)
            ai_result = generate_ai_result(result=tbs_result, roiid=roi['id'])
            ai_result.update(generate_dna_ai_result(result=dna_result, roiid=roi['id']))

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                mark_type=3,
                method='rectangle',
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result,
            ))

        ai_result = roi_marks[0].ai_result

        ai_suggest = f"{' '.join(ai_result['diagnosis'])} {','.join(ai_result['microbe'])};{ai_result['dna_diagnosis']}"
        return ALGResult(
            ai_suggest=ai_suggest,
            roi_marks=roi_marks,
            slide_quality=ai_result['slide_quality'],
            cell_num=ai_result['cell_num'],
            prob_dict=prob_dict
        )

    def run_dna_ploidy(self) -> ALGResult:

        rois = []

        from stone.modules.ai.libs.algorithms.DNA2.dna_alg import DNA_1020
        dna_alg_model = DNA_1020()

        slide = open_slide(task_param.slice_path)

        roi_marks = []
        prob_dict = None

        for idx, roi in enumerate(rois or [task_param.new_default_roi()]):
            dna_ploidy_result = dna_alg_model.dna_test(slide)

            from stone.modules.ai.utils.tct import generate_dna_ploidy_aiResult
            ai_result = generate_dna_ploidy_aiResult(result=dna_ploidy_result, roiid=roi['id'])

            roi_marks.append(Mark(
                id=roi['id'],
                position={'x': [], 'y': []},
                mark_type=3,
                method='rectangle',
                radius=5,
                is_export=1,
                stroke_color='grey',
                ai_result=ai_result,
            ))

        ai_result = roi_marks[0].ai_result

        return ALGResult(
            ai_suggest=ai_result['dna_diagnosis'],
            roi_marks=roi_marks,
            cell_num=ai_result['cell_num'],
            prob_dict=prob_dict
        )

    def run_her2(self, group_name_to_id: dict) -> ALGResult:
        cell_marks = []
        roi_marks = []
        ai_result = {}

        slide = open_slide(task_param.slice_path)
        mpp = slide.mpp or 0.242042

        rois = [] or [task_param.new_default_roi()]

        from stone.modules.ai.libs.algorithms.Her2New_.detect_all import run_her2_alg, roi_filter

        center_coords_np_with_id, cls_labels_np_with_id, summary_dict, lvl, flg = run_her2_alg(
            slice_path=task_param.slice_path, roi_list=rois)

        for roi in rois:
            roi_id, x_coords, y_coords = roi['id'], roi['x'], roi['y']
            center_coords, cls_labels = roi_filter(
                center_coords_np_with_id[roi_id],
                cls_labels_np_with_id[roi_id],
                x_coords,
                y_coords
            )

            ai_result = Her2Consts.rois_summary_dict.copy()
            label_to_roi_name = Her2Consts.cell_label_dict
            for idx, coord in enumerate(center_coords):
                roi_name = label_to_roi_name[str(cls_labels[idx])]
                ai_result[roi_name] += 1

                x = float(coord[0]) if slide.width > float(coord[0]) else float(slide.width - 1)
                y = float(coord[1]) if slide.height > float(coord[1]) else float(slide.height - 1)

                mark = Mark(
                    position={'x': [x], 'y': [y]},
                    fill_color=Her2Consts.type_color_dict[Her2Consts.label_to_diagnosis_type[int(cls_labels[idx])]],
                    mark_type=2,
                    diagnosis={'type': Her2Consts.label_to_diagnosis_type[int(cls_labels[idx])]},
                    radius=1 / mpp,
                    editable=0,
                    group_id=group_name_to_id.get(Her2Consts.idx_to_label[int(cls_labels[idx])]),
                    area_id=roi_id,
                    method='spot'
                )
                cell_marks.append(mark)

            whole_slide = 1 if len(x_coords) == 0 else 0
            group_id = group_name_to_id.get('ROI') if whole_slide != 1 else None

            ai_result.update({
                'whole_slide': whole_slide,
                '分级结果': Her2Consts.level[int(lvl)]
            })

            roi_marks.append(Mark(
                id=roi_id,
                position={'x': x_coords, 'y': y_coords},
                method='rectangle',
                mark_type=3,
                is_export=1,
                ai_result=ai_result,
                editable=1,
                stroke_color='grey',
                group_id=group_id
            ))

        return ALGResult(
            ai_suggest=ai_result['分级结果'],
            cell_marks=cell_marks,
            roi_marks=roi_marks,
        )

    @create_slice_db()
    def create_ai_marks(
            self,
            analysis_key: str,
            slice_key: str,
            ai_model: str,
            model_version: str,
            cell_marks: List[dict],
            roi_marks: List[dict],
            skip_mark_to_tile: bool = False
    ) -> Tuple[bool, Optional[List[MarkEntity]]]:
        cell_mark_entities, roi_mark_entities = [], []
        # group_ids = set()

        id_worker = IdWorker(1, 2, 0)

        for item in roi_marks + cell_marks:
            item['create_time'] = int(round(time.time() * 1000))
            if 'id' not in item:
                item['id'] = id_worker.get_next_id() or id_worker.get_new_id()
            # if not item['position']['x']:
            #     whole_slide_roi = self.repository.get_mark(item['id'])
            #     if whole_slide_roi:
            #         whole_slide_roi.update_data(**item)
            #     else:
            #         whole_slide_roi = MarkEntity(raw_data=item)
            #     self.repository.save_mark(whole_slide_roi)
            #     continue

            new_mark = MarkEntity(**item)

            if new_mark.mark_type == 3:
                roi_mark_entities.append(new_mark)
            else:
                cell_mark_entities.append(new_mark)

        saved = self.repository.batch_save_marks(roi_mark_entities) and self.repository.batch_save_marks(
            cell_mark_entities)

        if not saved:
            return False, None

        return True, cell_mark_entities + roi_mark_entities

    def get_analyses(self, **kwargs) -> Tuple[List[AnalysisVO], dict, str]:
        page = kwargs['page']
        per_page = kwargs['per_page']
        slice_id = kwargs['slice_id']
        userid = kwargs.get('userid')
        analyses, pagination = self.repository.get_analyses(page, per_page, slice_id, userid)
        analyses_hack = [AnalysisVO.parse_obj({**analysis.dict(), 'delete_permission': analysis.userid == 1}) for analysis in analyses]
        return analyses_hack, pagination, 'Get analyses success'

    def get_analysis(self, analysis_id: int) -> Tuple[Optional[AnalysisEntity], str]:
        analysis = self.repository.get_analysis_by_pk(analysis_id)
        return analysis, 'get analysis success'

    def create_analysis(self, **kwargs) -> Tuple[Optional[AnalysisEntity], str]:
        success, new_analysis = self.repository.save_analysis(AnalysisEntity.parse_obj(kwargs))
        if success:
            return new_analysis, 'create analysis success'
        return None, 'create analysis failed'

    def delete_analysis(self, analysis_id: int) -> Tuple[int, str]:
        analysis = self.repository.get_analysis_by_pk(analysis_id)

        deleted_count = self.repository.delete_analysis_by_pk(analysis_id)
        if deleted_count:
            fs.remove_dir(get_db_dir(analysis.slice_key, analysis.key))
            return deleted_count, 'Deleted analysis succeed'
        return deleted_count, 'Deleted analysis failed'

    @connect_slice_db()
    def get_marks(self, analysis_id: int) -> Tuple[List[MarkEntity], Optional[int], str]:
        marks, total = self.repository.get_marks()
        return marks, total, 'get marks success'
