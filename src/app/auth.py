import jwt
from sqlalchemy.exc import NoResultFound
from functools import wraps
from src.app.request_context import request_context


def auth1(token):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, "HS256")
        userid = payload["userid"]
        user = User.objects.get(pk=userid)
        user.token = token
        return user
    except (jwt.ExpiredSignatureError, jwt.DecodeError, NoResultFound):
        return None


def auth():
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            request_context.connect_db()
            return func(*args, **kwargs)
        return wrapped
    return deco
