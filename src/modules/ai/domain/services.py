from typing import Optional, Type

from src.libs.heimdall.dispatch import open_slide
from src.modules.ai.domain.value_objects import AITaskVO
from src.modules.ai.infrastructure.repositories import SQLAlchemyAiRepository


class AiDomainService(object):

    def __init__(self, repository: SQLAlchemyAiRepository):
        self.repository = repository

    def _select_model(self, ai_model, model_version) -> Optional[Type]:
        pass

    def run_tct(self, task: AITaskVO):
        slice_id = task.slice_id
        slide_path = task.slide_path
        ai_model = task.ai_model
        model_version = task.model_version

        alg_class = self._select_model(ai_model, model_version)

        slide = open_slide(task.slide_path)

