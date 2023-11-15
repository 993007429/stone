import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth
from src.app.db import connect_db
from src.app.request_context import request_context
from src.app.schema.user import PageQuery, UserIn, LoginIn
from src.app.service_factory import AppServiceFactory

user_blueprint = APIBlueprint('user', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@connect_db()
@user_blueprint.input(LoginIn, location='json')
def login(json_data):
    res = AppServiceFactory.user_service.login(**json_data)
    return res.dict


@user_blueprint.post('')
@connect_db()
@user_blueprint.input(UserIn, location='json')
def create_user(json_data):
    res = AppServiceFactory.user_service.create_user(**json_data)
    return res.dict


@user_blueprint.get('')
@connect_db()
@user_blueprint.input(PageQuery, location='query')
def get_users(query_data):
    res = AppServiceFactory.user_service.get_users()
    return res.dict
