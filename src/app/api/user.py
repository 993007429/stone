import asyncio
from typing import List

from apiflask import APIBlueprint
from marshmallow.fields import Integer

from src.app.auth import auth_required
from src.app.db import connect_db
from src.app.permission import permission_required
from src.app.request_context import request_context
from src.app.schema.user import PageQuery, UserIn, LoginIn, SingleUserOut, ListUserOut, ApiLoginOut
from src.app.service_factory import AppServiceFactory
from src.modules.user.infrastructure.permissions import IsAdmin

user_blueprint = APIBlueprint('用户模块', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@connect_db()
@user_blueprint.input(LoginIn, location='json')
@user_blueprint.output(ApiLoginOut)
@user_blueprint.doc(security='ApiAuth', summary='登录')
def login(json_data):
    res = AppServiceFactory.user_service.login(**json_data)
    return res.response


@user_blueprint.post('')
@connect_db()
@auth_required()
@permission_required([IsAdmin])
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(SingleUserOut)
@user_blueprint.doc(security='ApiAuth', summary='创建user')
def create_user(json_data):
    res = AppServiceFactory.user_service.create_user(**json_data)
    return res.response


@user_blueprint.get('')
@connect_db()
@auth_required()
@permission_required([IsAdmin])
@user_blueprint.input(PageQuery, location='query')
@user_blueprint.output(ListUserOut)
@user_blueprint.doc(security='ApiAuth', summary='user列表')
def get_users(query_data):
    res = AppServiceFactory.user_service.get_users()
    return res.response


@user_blueprint.get('/<int:userid>')
@connect_db()
# @auth_required()
# @permission_required([IsAdmin])
@user_blueprint.output(SingleUserOut)
@user_blueprint.doc(security='ApiAuth', summary='user详情')
def get_user(userid):
    res = AppServiceFactory.user_service.get_user(userid)
    return res.response


@user_blueprint.put('/<int:userid>')
@connect_db()
# @auth_required()
# @permission_required([IsAdmin])
@user_blueprint.output(SingleUserOut)
@user_blueprint.doc(security='ApiAuth', summary='更新user')
def update_user(userid):
    res = AppServiceFactory.user_service.update_user(userid)
    return res.response


@user_blueprint.delete('/<int:userid>')
@connect_db()
# @auth_required()
# @permission_required([IsAdmin])
@user_blueprint.output(SingleUserOut)
@user_blueprint.doc(security='ApiAuth', summary='删除user')
def delete_user(userid):
    res = AppServiceFactory.user_service.delete_user(userid)
    return res.response
