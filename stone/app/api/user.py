from apiflask import APIBlueprint

from stone.app.base_schema import APIAffectedCountOut
from stone.app.permission import permission_required
from stone.app.schema.user import UserPageQuery, UserIn, LoginIn, ApiLoginOut, ApiSingleUserOut, ApiListUserOut
from stone.app.service_factory import AppServiceFactory
from stone.modules.user.infrastructure.permissions import IsAdmin

user_blueprint = APIBlueprint('用户', __name__, url_prefix='/users')


@user_blueprint.post('/login')
@user_blueprint.input(LoginIn, location='json')
@user_blueprint.output(ApiLoginOut)
@user_blueprint.doc(summary='登录', security='ApiAuth')
def login(json_data):
    res = AppServiceFactory.user_service.login(**json_data)
    return res.response


@user_blueprint.post('')
@permission_required([IsAdmin])
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(ApiSingleUserOut)
@user_blueprint.doc(summary='创建user', security='ApiAuth')
def create_user(json_data):
    res = AppServiceFactory.user_service.create_user(**json_data)
    return res.response


@user_blueprint.get('')
@permission_required([IsAdmin])
@user_blueprint.input(UserPageQuery, location='query')
@user_blueprint.output(ApiListUserOut)
@user_blueprint.doc(summary='user列表', security='ApiAuth')
def get_users(query_data):
    res = AppServiceFactory.user_service.get_users(**query_data)
    return res.response


@user_blueprint.get('/<int:userid>')
@permission_required([IsAdmin])
@user_blueprint.output(ApiSingleUserOut)
@user_blueprint.doc(summary='user详情', security='ApiAuth')
def get_user(userid):
    res = AppServiceFactory.user_service.get_user(userid)
    return res.response


@user_blueprint.put('/<int:userid>')
@permission_required([IsAdmin])
@user_blueprint.input(UserIn, location='json')
@user_blueprint.output(ApiSingleUserOut)
@user_blueprint.doc(summary='更新user', security='ApiAuth')
def update_user(userid, json_data):
    res = AppServiceFactory.user_service.update_user(**{'user_id': userid, 'user_data': json_data})
    return res.response


@user_blueprint.delete('/<int:userid>')
@permission_required([IsAdmin])
@user_blueprint.output(APIAffectedCountOut)
@user_blueprint.doc(summary='删除user', security='ApiAuth')
def delete_user(userid):
    res = AppServiceFactory.user_service.delete_user(userid)
    return res.response
