from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.slice.domain.repositories import SliceRepository


class SQLAlchemySliceRepository(SliceRepository):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

