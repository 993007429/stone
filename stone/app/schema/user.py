from apiflask import Schema
from marshmallow.fields import Integer, String, List, Nested, DateTime
from marshmallow.validate import Length, OneOf

from stone.app.base_schema import PageQuery, PaginationSchema
from stone.modules.user.domain.value_objects import RoleType


class UserPageQuery(PageQuery):
    pass


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 255)])
    password = String(required=True, validate=[Length(8, 32)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])


class LoginIn(Schema):
    username = String(required=True, validate=[Length(0, 100)])
    password = String(required=True, validate=[Length(8, 32)])


class UserOut(Schema):
    id = Integer(required=True)
    username = String(required=True)
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    creator = String(required=True)
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')


class SingleUserOut(Schema):
    user = Nested(UserOut())


class ApiSingleUserOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleUserOut())


class ListUserOut(Schema):
    users = List(Nested(UserOut()))


class ApiListUserOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(ListUserOut())
    pagination = Nested(PaginationSchema())


class LoginOut(Schema):
    userid = Integer(required=True)
    username = String(required=True)
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    token = String(required=True)


class SingleLoginOut(Schema):
    login = Nested(LoginOut())


class ApiLoginOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(SingleLoginOut())
