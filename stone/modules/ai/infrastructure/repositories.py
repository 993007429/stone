from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from stone.modules.ai.domain.entities import MarkEntity, AnalysisEntity
from stone.modules.ai.domain.repositories import AIRepository
from stone.modules.ai.domain.value_objects import AIType
from stone.modules.ai.infrastructure.mark_models import get_ai_mark_model, get_ai_mark_to_tile_model, NPCountModel, \
    Pdl1sCountModel, MarkGroupModel, ChangeRecordModel
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
        self._session.begin()
        self._session.add(model)
        self._session.flush([model])
        self._session.commit()
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
        self._slice_db_session.begin()
        self._slice_db_session.add(model)
        self._slice_db_session.flush([model])
        self._slice_db_session.commit()
        return True, 'Create mark success'

    def batch_save_marks(self, entities: List[MarkEntity]) -> bool:
        self._slice_db_session.begin()
        self._slice_db_session.bulk_insert_mappings(self.mark_model_class, [entity.dict() for entity in entities])
        self._slice_db_session.commit()
        return True

    def create_mark_tables(self, ai_type: AIType):
        if not self.mark_table_suffix:
            return

        engine = self._slice_db_session.get_bind()
        tables = [
            get_ai_mark_model(self.mark_table_suffix).__table__,
            get_ai_mark_to_tile_model(self.mark_table_suffix).__table__,
        ]

        if ai_type == AIType.np:
            tables.append(NPCountModel.__table__)
        elif ai_type == AIType.pdl1:
            tables.append(Pdl1sCountModel.__table__)

        tables.append(MarkGroupModel.__table__)
        tables.append(ChangeRecordModel.__table__)

        Base.metadata.create_all(engine, tables=tables)

    def get_analyses(self, **kwargs) -> List[AnalysisEntity]:
        self._session.begin()
        query = self._session.query(Analysis).filter_by(**kwargs)
        models = query.all()
        self._session.commit()
        return [AnalysisEntity(**model.dict) for model in models]

    def get_analysis_by_pk(self, pk: int) -> Optional[AnalysisEntity]:
        self._session.begin()
        query = self._session.query(Analysis).filter_by(id=pk)
        model = query.first()
        self._session.commit()
        if not model:
            return None
        return AnalysisEntity(**model.dict)










