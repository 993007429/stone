from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.ai.domain.entities import MarkEntity
from src.modules.ai.domain.repositories import AIRepository
from src.modules.ai.domain.value_objects import AIType
from src.modules.ai.infrastructure.models import get_ai_mark_model, get_ai_mark_to_tile_model, NPCountModel, \
    Pdl1sCountModel, MarkGroupModel, ChangeRecordModel
from src.seedwork.infrastructure.models import Base


class SQLAlchemyAIRepository(AIRepository):

    def __init__(self, session: ContextVar, slice_db_session: ContextVar):
        self._session_cv = session
        self._slice_db_session_cv = slice_db_session
        self._mark_table_suffix = None

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    @property
    def mark_table_suffix(self):
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
        self._slice_db_session.bulk_insert_mappings(self.mark_model_class, [entity.dict() for entity in entities])
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












