from apiflask import Schema, PaginationSchema
from apiflask.fields import Integer, String, List, Nested, DateTime
from apiflask.validators import Range
from apiflask.validators import Length, OneOf
from marshmallow import post_dump

from src.modules.user.domain.value_objects import RoleType


class UserIn(Schema):
    username = String(required=True, validate=[Length(0, 255)])
    password = String(required=True, validate=[Length(8, 32)])
    role = String(required=True, validate=[OneOf([RoleType.admin.value, RoleType.user.value])])
    creator = String(required=True, validate=[Length(0, 255)])


class PageQuery(Schema):
    page = Integer(load_default=1)
    per_page = Integer(load_default=10, validate=Range(max=100))


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


class ApiUserOut(Schema):
    http_code = Integer(required=True)
    err_code = Integer(required=True)
    message = String(required=True)
    data = Nested(UserOut)
