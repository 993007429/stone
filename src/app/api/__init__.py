from apiflask import APIBlueprint

from src.app.api.ai import ai_blueprint
from src.app.api.user import user_blueprint

api_blueprint = APIBlueprint('api', __name__, url_prefix='/api')
api_blueprint.register_blueprint(ai_blueprint)
api_blueprint.register_blueprint(user_blueprint)
