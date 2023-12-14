from apiflask import APIBlueprint
from flask import request

import setting
from stone.app.api.ai import ai_blueprint
from stone.app.api.analysis import analysis_blueprint
from stone.app.api.dataset import dataset_blueprint
from stone.app.api.ft import filter_template_blueprint
from stone.app.api.label import label_blueprint
from stone.app.api.slice import slice_blueprint
from stone.app.api.user import user_blueprint
from stone.app.auth import auth_token
from stone.app.request_context import request_context

api_blueprint = APIBlueprint('stone', __name__, url_prefix='/api')

api_blueprint.register_blueprint(user_blueprint)
api_blueprint.register_blueprint(slice_blueprint)
api_blueprint.register_blueprint(ai_blueprint)
api_blueprint.register_blueprint(analysis_blueprint)
api_blueprint.register_blueprint(dataset_blueprint)
api_blueprint.register_blueprint(label_blueprint)
api_blueprint.register_blueprint(filter_template_blueprint)


def api_before_request():
    request_context.host_url = request.host_url
    request_context.connect_db()
    if request.path not in setting.WHITE_LIST:
        token = request.headers.get("Authorization")
        # return auth_token(token)


def api_after_request(response):
    if response.status_code >= 400:
        request_context.close_db(commit=False)
    else:
        request_context.close_db()
    return response


api_blueprint.before_request(api_before_request)
api_blueprint.after_request(api_after_request)
