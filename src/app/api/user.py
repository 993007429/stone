import asyncio
from apiflask import APIBlueprint
from flask import request
from marshmallow.fields import Integer

from src.app.schema.user import PetsOut, PetQuery

user_blueprint = APIBlueprint('user', __name__, url_prefix='/user')


@user_blueprint.post('/login')
@user_blueprint.input(PetQuery, location='query')
@user_blueprint.output({'answer': Integer(dump_default=42)})
def login(query_data):
    return {'answer': 1}
