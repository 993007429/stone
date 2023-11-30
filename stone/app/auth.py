import jwt
from flask import request
from sqlalchemy.exc import NoResultFound
from functools import wraps

import setting
from stone.app.request_context import request_context
from stone.app.service_factory import AppServiceFactory


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
