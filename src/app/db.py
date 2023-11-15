from functools import wraps

from src.app.request_context import request_context


def connect_db():
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            request_context.connect_db()
            return func(*args, **kwargs)
        return wrapped
    return deco
