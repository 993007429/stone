from functools import wraps

from src.app.request_context import request_context


def connect_db():
    def deco(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            request_context.connect_db()
            r = f(*args, **kwargs)
            return r
        return wrapped
    return deco
