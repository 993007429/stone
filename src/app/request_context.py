from contextvars import ContextVar

from sqlalchemy.orm import Session

from src.infra.session import get_session


class RequestContext:
    _db_session: ContextVar[Session] = ContextVar("_db_session", default=None)

    @property
    def db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._db_session

    def connect_db(self):
        session = get_session()
        self._db_session.set(session)


request_context = RequestContext()
