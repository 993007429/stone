from apiflask import APIFlask
from flask import jsonify

from src.app.api import api_blueprint

app = APIFlask(__name__)
app.config.from_object('setting')

app.register_blueprint(api_blueprint)
