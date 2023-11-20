from apiflask import Schema
from apiflask.fields import Integer, String, List, Nested, DateTime
from apiflask.validators import Range
from apiflask.validators import Length, OneOf

from src.app.base_schema import PageQuery, PaginationSchema
from src.modules.user.domain.value_objects import RoleType


class UserPageQuery(PageQuery):
    pass


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 255)])
    password = String(required=True, validate=[Length(8, 32)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    creator = String(required=True, validate=[Length(0, 255)])


class LoginIn(Schema):
    username = String(required=True, validate=[Length(0, 100)])
    password = String(required=True, validate=[Length(8, 32)])


class UserOut(Schema):
    id = Integer(required=True)
    username = String(required=True, validate=[Length(0, 255)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    creator = String(required=True, validate=[Length(0, 255)])
    created_at = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')
    last_modified = DateTime(required=True, format='%Y-%m-%d %H:%M:%S')


class SingleUserOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(UserOut)


class ListUserOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = List(Nested(UserOut))
    pagination = Nested(PaginationSchema)


class LoginOut(Schema):
    userid = Integer(required=True)
    username = String(required=True, validate=[Length(0, 255)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    token = String(required=True)


class ApiLoginOut(Schema):
    code = Integer(required=True)
    message = String(required=True)
    data = Nested(LoginOut)
