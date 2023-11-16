import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import token_required
from src.app.db import connect_db
from src.app.request_context import request_context
from src.app.schema.user import PageQuery, UserIn, LoginIn, ApiUserOut
from src.app.service_factory import AppServiceFactory

user_blueprint = APIBlueprint('user', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@connect_db()
@user_blueprint.input(LoginIn, location='json')
def login(json_data):
    res = AppServiceFactory.user_service.login(**json_data)
    return res.response


@user_blueprint.post('')
@connect_db()
# @token_required()
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(ApiUserOut)
@user_blueprint.doc(security='ApiAuth')
def create_user(json_data):
    res = AppServiceFactory.user_service.create_user(**json_data)
    return res.response


@user_blueprint.get('')
@connect_db()
@token_required()
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.doc(security='ApiAuth')
def get_users(query_data):
    res = AppServiceFactory.user_service.get_users()
    return res.response
