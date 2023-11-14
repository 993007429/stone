import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth
from src.app.decorators import connect_db
from src.app.request_context import request_context
from src.app.schema.user import PageQuery, UserIn, UserOut, UsersOut
from src.app.service_factory import AppServiceFactory

user_blueprint = APIBlueprint('user', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.output({'answer': Integer(dump_default=42)})
def login(query_data):
    return {'answer': 1}


@user_blueprint.auth_required(auth)
@user_blueprint.post('')
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(UserOut)
@user_blueprint.doc(security='ApiKeyAuth')
def create_user(json_data):
    res = AppServiceFactory.user_service.create_user(**json_data)
    return res.dict()


@connect_db()
@user_blueprint.auth_required(auth)
@user_blueprint.get('')
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.output(UsersOut)
def get_users(query_data):
    res = AppServiceFactory.user_service.get_users()
    return res.dict()
