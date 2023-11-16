from apiflask import APIFlask
from flask import jsonify

from src.app.api import api_blueprint

app = APIFlask(__name__)
app.config.from_object('setting')

app.register_blueprint(api_blueprint)


@app.after_request
def after_request(response):
    res = response.json
    return jsonify(
        code=res.get('err_code'),
        message=res.get('message'),
        data=res.get('data')
    )
