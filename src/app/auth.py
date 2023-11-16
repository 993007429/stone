import jwt
from flask import request
from sqlalchemy.exc import NoResultFound
from functools import wraps

import setting
from src.app.request_context import request_context
from src.app.service_factory import AppServiceFactory


def token_required():
    def deco(func):
        @wraps(func)
        def wrapped(*args, **kwargs):

            token = request.headers.get("Authorization")
            if not token:
                return {'code': 401, 'message': 'Token is missing'}

            try:
                payload = jwt.decode(token, setting.SECRET_KEY, "HS256")
                userid = payload["userid"]
                user = AppServiceFactory.user_service.domain_service.repository.get_user_by_pk(userid)
                if not user:
                    return {'code': 401, 'message': 'Invalid Token'}

                return func(*args, **kwargs)

            except jwt.ExpiredSignatureError:
                return {'code': 401, 'message': 'Token has expired'}

            except (jwt.InvalidTokenError, jwt.DecodeError) as e:
                return {'code': 401, 'message': 'Invalid Token'}

        return wrapped
    return deco
