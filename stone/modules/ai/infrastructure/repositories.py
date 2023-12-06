from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.modules.ai.domain.entities import MarkEntity, AnalysisEntity
from stone.modules.ai.domain.repositories import AIRepository
from stone.modules.ai.domain.enum import AIModel
from stone.modules.ai.infrastructure.mark_models import get_ai_mark_model, get_ai_mark_to_tile_model, NPCountModel, Pdl1sCountModel, MarkGroupModel, ChangeRecordModel
from stone.modules.ai.infrastructure.models import Analysis
from stone.seedwork.infrastructure.models import Base


class SQLAlchemyAIRepository(AIRepository):

    def __init__(self, session: ContextVar, slice_db_session: ContextVar, manual: Optional[AIRepository] = None):
        self._session_cv = session
        self._slice_db_session_cv = slice_db_session
        self._mark_table_suffix = None
        self._manual = manual

    @property
    def manual(self) -> 'AIRepository':
        return self._manual

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    def save(self, entity: AnalysisEntity) -> bool:
        model = Analysis(**entity.dict())
        self._session.add(model)
        self._session.flush([model])
        entity.from_orm(model)
        return True

    @property
    def mark_table_suffix(self) -> Optional[str]:
        return self._mark_table_suffix

    @mark_table_suffix.setter
    def mark_table_suffix(self, value):
        self._mark_table_suffix = value

    @property
    def _slice_db_session(self) -> Session:
        s = self._slice_db_session_cv.get()
        assert s is not None
        return s

    @property
    def mark_model_class(self):
        return get_ai_mark_model(self.mark_table_suffix)

    def save_mark(self, entity: MarkEntity) -> Tuple[bool, str]:
        model = self.mark_model_class(**entity.dict())
        self._slice_db_session.add(model)
        self._slice_db_session.flush([model])
        return True, 'Create mark success'

    def batch_save_marks(self, entities: List[MarkEntity]) -> bool:
        self._slice_db_session.bulk_insert_mappings(self.mark_model_class, [entity.dict() for entity in entities])
        return True

    def create_mark_tables(self, ai_model: str):
        if not self.mark_table_suffix:
            return

        engine = self._slice_db_session.get_bind()
        tables = [get_ai_mark_model(self.mark_table_suffix).__table__]

        if ai_model not in [AIModel.tct1, AIModel.tct2, AIModel.lct1, AIModel.lct2, AIModel.dna, AIModel.dna_ploidy]:
            tables.append(get_ai_mark_to_tile_model(self.mark_table_suffix).__table__)
        if ai_model == AIModel.np:
            tables.append(NPCountModel.__table__)
        if ai_model == AIModel.pdl1:
            tables.append(Pdl1sCountModel.__table__)

        Base.metadata.create_all(engine, tables=tables)

    def get_analyses(self, **kwargs) -> List[AnalysisEntity]:
        query = self._session.query(Analysis).filter_by(**kwargs)
        models = query.all()
        return [AnalysisEntity.from_orm(model) for model in models]

    def get_analysis_by_pk(self, pk: int) -> Optional[AnalysisEntity]:
        query = self._session.query(Analysis).filter_by(id=pk)
        model = query.first()
        if not model:
            return None
        return AnalysisEntity.from_orm(model)

    def save_analysis1(self, entity: AnalysisEntity) -> Tuple[bool, Optional[AnalysisEntity]]:
        model = Analysis(**entity.dict())
        try:
            self._session.add(model)
            self._session.flush([model])
        except IntegrityError:
            return False, None
        return True, entity.from_orm(model)

    def save_analysis(self, entity: AnalysisEntity) -> Tuple[bool, Optional[AnalysisEntity]]:
        model = Analysis(**entity.dict())
        self._session.add(model)
        self._session.flush([model])
        return True, entity.from_orm(model)
