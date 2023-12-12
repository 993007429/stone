import math
from typing import List, Optional, Tuple, Type

from stone.modules.ai.domain.entities import MarkEntity, AnalysisEntity
from stone.modules.ai.domain.repositories import MarkRepository, AnalysisRepository
from stone.modules.ai.domain.enum import AIModel
from stone.modules.ai.infrastructure.mark_models import get_ai_mark_model, get_ai_mark_to_tile_model, NPCountModel, Pdl1sCountModel, MarkGroupModel, ChangeRecordModel
from stone.modules.ai.infrastructure.models import Analysis
from stone.seedwork.infrastructure.models import Base
from stone.seedwork.infrastructure.repositories import SQLAlchemySingleModelRepository


class SQLAlchemyMarkRepository(MarkRepository, SQLAlchemySingleModelRepository[MarkEntity]):

    def __init__(self, manual: Optional[MarkRepository] = None, **kwargs):
        super().__init__(**kwargs)
        self._mark_table_suffix = None
        self._manual = manual

    @property
    def model_class(self) -> Type[Analysis]:
        return get_ai_mark_model(self.mark_table_suffix)

    @property
    def manual(self) -> 'MarkRepository':
        return self._manual

    @property
    def mark_table_suffix(self) -> Optional[str]:
        return self._mark_table_suffix

    @mark_table_suffix.setter
    def mark_table_suffix(self, value):
        self._mark_table_suffix = value

    def create_mark_tables(self, ai_model: str):
        if not self.mark_table_suffix:
            return

        engine = self.session.get_bind()
        tables = [get_ai_mark_model(self.mark_table_suffix).__table__]

        if ai_model not in [AIModel.tct1, AIModel.tct2, AIModel.lct1, AIModel.lct2, AIModel.dna, AIModel.dna_ploidy]:
            tables.append(get_ai_mark_to_tile_model(self.mark_table_suffix).__table__)
        if ai_model == AIModel.np:
            tables.append(NPCountModel.__table__)
        if ai_model == AIModel.pdl1:
            tables.append(Pdl1sCountModel.__table__)

        Base.metadata.create_all(engine, tables=tables)


class SQLAlchemyAnalysisRepository(AnalysisRepository, SQLAlchemySingleModelRepository[AnalysisEntity]):

    @property
    def model_class(self) -> Type[Analysis]:
        return Analysis

    def get_analyses(self, page: int, per_page: int, slice_id: int, userid: Optional[int]) -> Tuple[List[AnalysisEntity], dict]:
        query = self.session.query(Analysis).filter(Analysis.slice_id == slice_id)
        if userid:
            query = query.filter(Analysis.userid == userid)
        total = query.count()
        query = query.order_by(Analysis.id)

        offset = min((page - 1), math.floor(total / per_page)) * per_page
        query = query.offset(offset).limit(per_page)
        pagination = {'total': total, 'page': page, 'per_page': per_page}

        analyses = [AnalysisEntity.from_orm(model) for model in query.all()]
        return analyses, pagination
