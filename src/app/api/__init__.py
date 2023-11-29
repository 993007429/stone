import jwt
from apiflask import APIBlueprint
from flask import request

import setting
from src.app.api.ai import ai_blueprint
from src.app.api.analysis import analysis_blueprint
from src.app.api.dataset import dataset_blueprint
from src.app.api.ft import ft_blueprint
from src.app.api.label import label_blueprint
from src.app.api.slice import slice_blueprint
from src.app.api.user import user_blueprint
from src.app.request_context import request_context
from src.modules.user.domain.value_objects import LoginUser

api_blueprint = APIBlueprint('stone', __name__, url_prefix='/api')

api_blueprint.register_blueprint(user_blueprint)
api_blueprint.register_blueprint(slice_blueprint)
api_blueprint.register_blueprint(ai_blueprint)
api_blueprint.register_blueprint(analysis_blueprint)
api_blueprint.register_blueprint(dataset_blueprint)
api_blueprint.register_blueprint(label_blueprint)
api_blueprint.register_blueprint(ft_blueprint)


def api_before_request():
    request_context.connect_db()
    token = request.headers.get("Authorization")
    if token:
        try:
            payload = jwt.decode(token, setting.SECRET_KEY, "HS256")
            userid, username, role = payload["userid"], payload["username"], payload["role"]

        except jwt.ExpiredSignatureError:
            return {'code': 401, 'message': 'Token has expired'}

        except (jwt.InvalidTokenError, jwt.DecodeError) as e:
            return {'code': 401, 'message': 'Invalid Token'}

        request_context.token = token
        request_context.current_user = LoginUser(userid=userid, username=username, role=role, token=token)


def api_after_request(response):
    request_context.close_db(commit=True)
    return response


api_blueprint.before_request(api_before_request)
api_blueprint.after_request(api_after_request)
