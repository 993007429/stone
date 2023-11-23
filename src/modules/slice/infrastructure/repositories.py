from contextvars import ContextVar
from typing import List, Optional, Tuple

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.modules.slice.domain.entities import SliceEntity
from src.modules.slice.domain.repositories import SliceRepository
from src.modules.slice.infrastructure.models import Slice


class SQLAlchemySliceRepository(SliceRepository):

    def __init__(self, session: ContextVar):
        self._session_cv = session

    @property
    def _session(self) -> Session:
        s = self._session_cv.get()
        assert s is not None
        return s

    def get_slice_by_id(self, pk: int) -> Optional[SliceEntity]:
        self._session.begin()
        query = self._session.query(Slice).filter_by(id=pk)
        model = query.first()
        self._session.commit()
        if not model:
            return None
        return SliceEntity(**model.dict)
