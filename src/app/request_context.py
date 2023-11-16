from contextvars import ContextVar

from sqlalchemy.orm import Session

from src.infra.session import get_session
from src.modules.user.domain.value_objects import LoginUser


class RequestContext:
    _db_session: ContextVar[Session] = ContextVar('_db_session', default=None)
    _token: ContextVar = ContextVar('_token', default=None)
    _current_user: ContextVar = ContextVar("_current_user", default=None)

    @property
    def db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._db_session

    def connect_db(self):
        session = get_session()
        self._db_session.set(session)

    @property
    def token(self):
        return self._token.get()

    @token.setter
    def token(self, value):
        self._token.set(value)

    @property
    def current_user(self) -> LoginUser:
        return self._current_user.get()

    @current_user.setter
    def current_user(self, user: LoginUser):
        self._current_user.set(user)


request_context = RequestContext()
