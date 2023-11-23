from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.ai.domain.repositories import AnalysisRepository


class SQLAlchemyAiRepository(AnalysisRepository):

    def __init__(self, session: ContextVar, slice_db_session: ContextVar):
        self._session_cv = session
        self._slice_db_session_cv = slice_db_session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    @property
    def _slice_db_session(self) -> Session:
        s = self._slice_db_session_cv.get()
        assert s is not None
        return s

