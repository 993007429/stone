from contextvars import ContextVar

from sqlalchemy.orm import Session

from stone.infra.session import get_session, get_session_by_db_uri
from stone.modules.user.domain.value_objects import LoginUser


class RequestContext:
    _db_session: ContextVar[Session] = ContextVar('_db_session', default=None)
    _slice_db_session: ContextVar[Session] = ContextVar('_slice_db_session', default=None)
    _token: ContextVar = ContextVar('_token', default=None)
    _current_user: ContextVar = ContextVar('_current_user', default=None)
    _host_url: ContextVar = ContextVar('_host_url', default='')

    @property
    def db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._db_session

    @property
    def slice_db_session(self) -> ContextVar[Session]:
        """Get current db session as ContextVar"""
        return self._slice_db_session

    def connect_db(self):
        session = get_session()
        session.begin()
        self.db_session.set(session)

    def close_db(self, commit: bool = True):
        session = self.db_session.get()
        if session and session.is_active:
            if commit:
                session.commit()
            else:
                session.rollback()
            session.close()

    def connect_slice_db(self, db_file_path: str):
        session = get_session_by_db_uri(f'sqlite:///{db_file_path}')
        session.begin()
        self.slice_db_session.set(session)

    def close_slice_db(self, commit: bool = True):
        session = self.slice_db_session.get()
        if session and session.is_active:
            if commit:
                session.commit()
            else:
                session.rollback()
            session.close()

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

    @property
    def host_url(self):
        return self._host_url.get()

    @host_url.setter
    def host_url(self, value):
        self._host_url.set(value)


request_context = RequestContext()
