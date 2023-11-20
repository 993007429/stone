from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.analysis.domain.repositories import AnalysisRepository


class SQLAlchemyAnalysisRepository(AnalysisRepository):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

