import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.schema.user import PageQuery, UserIn, UserOut, UsersOut
from src.modules.user.application.services import UserService
from src.modules.user.infrastructure.repositories import SQLAlchemyUserRepository
from src.modules.user.domain.services import UserDomainService

user_blueprint = APIBlueprint('user', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.output({'answer': Integer(dump_default=42)})
def login(query_data):
    return {'answer': 1}


@user_blueprint.post('')
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(UserOut)
def create_user(json_data):
    res = UserService(UserDomainService(SQLAlchemyUserRepository())).create_user(**json_data)
    return res.dict()


@user_blueprint.get('')
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.output(UsersOut)
def get_users(query_data):
    res = UserService(UserDomainService(SQLAlchemyUserRepository())).get_users()
    return res.dict()
