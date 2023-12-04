import jwt
from functools import wraps

import setting
from stone.app.request_context import request_context
from stone.app.service_factory import AppServiceFactory
from stone.modules.user.domain.value_objects import LoginUser


def auth_required():
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):

            token = request_context.token
            if not token:
                return {'code': 401, 'message': 'Token is missing'}

            payload = jwt.decode(token, setting.SECRET_KEY, "HS256")
            userid = payload["userid"]
            user = AppServiceFactory.user_service.domain_service.repository.get_user_by_pk(userid)
            if not user:
                return {'code': 401, 'message': 'Invalid Token'}

            return func(*args, **kwargs)

        return wrapped
    return deco


def auth_token(token):
    if not token:
        return {'code': 401, 'message': 'Token is missing'}

    try:
        payload = jwt.decode(token, setting.SECRET_KEY, "HS256")
        userid, username, role = payload["userid"], payload["username"], payload["role"]

    except jwt.ExpiredSignatureError:
        return {'code': 401, 'message': 'Token has expired'}

    except (jwt.InvalidTokenError, jwt.DecodeError):
        return {'code': 401, 'message': 'Invalid Token'}

    login_user = AppServiceFactory.user_service.domain_service.repository.get_user_by_pk(userid)

    if not login_user:
        return {'code': 401, 'message': 'Invalid Token'}

    request_context.token = token
    request_context.current_user = LoginUser(userid=userid, username=username, role=role, token=token)
