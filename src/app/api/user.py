import asyncio
from apiflask import APIBlueprint
from flask import request
from marshmallow.fields import Integer

from src.app.schema.user import PetsOut, PetQuery, UserIn, UserOut
from src.modules.user.application.services import UserService
from src.modules.user.domain.repositories import UserRepository
from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository
from src.modules.user.domain.services import UserDomainService

user_blueprint = APIBlueprint('user', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@user_blueprint.input(PetQuery, location='query')
@user_blueprint.output({'answer': Integer(dump_default=42)})
def login(query_data):
    return {'answer': 1}


@user_blueprint.post('')
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(UserOut)
def create_user(json_data):
    username = json_data['username']
    password = json_data['password']

    res = UserService(UserDomainService(SQLAlchemyUserRepository())).create_user(username=username, password=password)
    return res.dict()


@user_blueprint.get('')
def list_users(json):
    pass
