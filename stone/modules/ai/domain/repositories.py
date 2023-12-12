from abc import ABCMeta
from typing import Optional, List, Tuple

from stone.modules.ai.domain.entities import AnalysisEntity, MarkEntity
from stone.seedwork.domain.repositories import SingleModelRepository


class MarkRepository(SingleModelRepository[MarkEntity], metaclass=ABCMeta):

    def create_mark_tables(self, ai_model: str):
        ...


class AnalysisRepository(SingleModelRepository[AnalysisEntity], metaclass=ABCMeta):

    def get_analyses(self, page: int, per_page: int, slice_id: int, userid: Optional[int]) -> Tuple[List[AnalysisEntity], dict]:
        ...
