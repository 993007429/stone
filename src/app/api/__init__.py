from flask import Blueprint

from src.app.api.ai import ai_blueprint

api_blueprint = Blueprint('api', __name__)
api_blueprint.register_blueprint(ai_blueprint)
